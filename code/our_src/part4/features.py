"""Frozen provided backbones; content-addressed per-image feature caches."""
import importlib.metadata
from pathlib import Path
import platform
import re
import time
import numpy as np
from PIL import Image, ImageOps
from .common import digest, file_hash, image_path, l2, npz, provided, read, save

DIMENSIONS = {'resnet18': {'layer2': 128, 'layer3': 256, 'layer4': 512},
              'dinov2_vits14': {'global': 384, 'patch_mean': 384, 'spatial2x2': 1536}, 'clip_vit_b_32': {'global': 512}}


def patch_grid(patches, input_hw, patch_size):
    """Recover row-major tokens from actual input size; reject padding assumptions."""
    ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    h, w = input_hw
    if h % ph or w % pw or patches.shape[1] != (h // ph) * (w // pw):
        raise ValueError('DINO patch count does not match input dimensions / patch size')
    return patches.reshape(patches.shape[0], h // ph, w // pw, patches.shape[-1])


def pool_patches(patches, input_hw, patch_size):
    import torch
    grid = patch_grid(patches, input_hw, patch_size)
    h, w = grid.shape[1:3]
    if min(h, w) < 2:
        raise ValueError('Spatial pooling requires at least two rows and columns')
    regions = [grid[:, y0:y1, x0:x1].mean(dim=(1, 2))
               for y0, y1 in ((0, h // 2), (h // 2, h))
               for x0, x1 in ((0, w // 2), (w // 2, w))]
    return {'patch_mean': patches.mean(dim=1), 'spatial2x2': torch.cat(regions, dim=-1)}


def preprocessing_description(preprocess):
    """Keep transform settings, excluding process-local Python function addresses."""
    return re.sub(r'(<function [^\n]*?) at 0x[0-9a-fA-F]+>', r'\1>', repr(preprocess))


class Backbone:
    def __init__(self, project, name, config):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('PyTorch is missing from this Python environment; install the project deep-model requirements.') from exc
        self.torch, self.name = torch, name
        self.device = config['device']
        torch.manual_seed(config['seed'])
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
            torch.backends.cudnn.benchmark = False
        try:
            self.model, self.preprocess = provided(project).getModel(name, device=self.device)
        except Exception as exc:
            raise RuntimeError(f'Cannot load pretrained {name}: {exc}. Check dependencies, official checkpoint cache and network access; no fallback weights are used.') from exc
        self.model.eval()
        self.model.requires_grad_(False)
        if name == 'dinov2_vits14':
            # Official dinov2/data/transforms.py evaluation defaults. The provided
            # loader uses bilinear; correct only our adapter, preserving src/.
            from torchvision import transforms
            self.preprocess = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        self.extractor = provided(project).IntermediateFeatureExtractor(self.model) if name == 'resnet18' else None
        self.metadata = dict(model=name, preprocessing=preprocessing_description(self.preprocess), device=self.device,
            hardware=platform.processor(), torch=torch.__version__,
            torchvision=importlib.metadata.version('torchvision'),
            loader_sha256=file_hash(Path(project)/'src/deep_models.py'),
            weights={'resnet18':'ResNet18_Weights.DEFAULT', 'dinov2_vits14':'official dinov2_vits14 pretrained',
                     'clip_vit_b_32':'OpenAI CLIP ViT-B/32'}[name])
        self.metadata['adapter_sha256'] = file_hash(__file__)
        self.metadata['cpu_threads'] = torch.get_num_threads()
        self.metadata['gpu'] = torch.cuda.get_device_name(self.device) if str(self.device).startswith('cuda') else None
        # Hash the actual loaded state, not merely a potentially moving model alias.
        import hashlib
        h = hashlib.sha256()
        for key, value in self.model.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.metadata['weights_sha256'] = h.hexdigest()

    def sync(self):
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize()

    def tensors(self, batch):
        torch = self.torch
        if self.name == 'resnet18':
            _, blocks = self.extractor(batch)
            return {name: blocks[name + '.1'] for name in DIMENSIONS[self.name]}
        if self.name == 'clip_vit_b_32':
            return {'global': self.model.encode_image(batch).float()}
        values = self.model.forward_features(batch)
        return {
            'global': values['x_norm_clstoken'],
            **pool_patches(values['x_norm_patchtokens'], batch.shape[-2:], self.model.patch_embed.patch_size)}

    def patch_similarity(self, image):
        """Patch-CLS cosine on the preprocessed crop; not an SVM explanation."""
        batch = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            values = self.model.forward_features(batch)
            grid = patch_grid(values['x_norm_patchtokens'], batch.shape[-2:], self.model.patch_embed.patch_size)
            similarity = self.torch.nn.functional.cosine_similarity(
                grid, values['x_norm_clstoken'][:, None, None, :], dim=-1)
        return similarity[0].cpu().numpy()

    def encode(self, images):
        start = time.perf_counter()
        batch = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
        self.sync()
        preprocessing = time.perf_counter() - start
        start = time.perf_counter()
        with self.torch.inference_mode():
            values = self.tensors(batch)
        self.sync()
        inference = time.perf_counter() - start
        start = time.perf_counter()
        if self.name == 'resnet18':
            with self.torch.inference_mode():
                values = {k:v.mean(dim=(-2,-1)) for k,v in values.items()}
        values = {k: v.float().cpu().numpy() for k, v in values.items()}
        return values, dict(preprocessing_seconds=preprocessing, backbone_seconds=inference,
                            encoding_seconds=time.perf_counter()-start)

    def close(self):
        if self.extractor:
            self.extractor.remove_hooks()


def extract(backbone, rows, project, config, root, flip=False):
    active = [r for r in rows if not flip or r['split'] == 'train']
    identity = digest(dict(metadata=backbone.metadata, augmentation='horizontal_flip' if flip else 'none',
                           batch_size=config['batch_size']))
    output = Path(root)/'features'/backbone.name/identity
    output.mkdir(parents=True, exist_ok=True)
    previous = read(output/'manifest.json') if (output/'manifest.json').exists() else {}
    files, missing = {}, []
    for row in active:
        path = image_path(project, config, row['image_id'])
        key = digest(dict(image_sha256=file_hash(path), row=row, identity=identity))
        destination = output/(key + '.npz')
        files[row['image_id']] = destination
        if destination.exists():
            expected = previous.get('file_sha256',{}).get(str(destination.relative_to(root)))
            if expected is not None and file_hash(destination)!=expected:
                raise ValueError(f'Feature cache checksum changed: {destination}')
            try:
                with np.load(destination, allow_pickle=False) as z:
                    for layer, d in DIMENSIONS[backbone.name].items():
                        if (str(z['image_id']) != row['image_id'] or str(z['split']) != row['split'] or
                            z[layer].shape != (d,) or not np.isfinite(z[layer]).all() or
                            z[layer+'_l2'].shape != (d,) or not np.isfinite(z[layer+'_l2']).all() or
                            not np.allclose(z[layer+'_l2'], l2(z[layer]), atol=1e-6)):
                            raise ValueError(f'Invalid cached descriptor {destination}')
            except (ValueError, OSError, KeyError) as exc:
                raise ValueError(f'Corrupt feature cache; preserve and inspect {destination}') from exc
        else:
            missing.append(row)
    for offset in range(0, len(missing), config['batch_size']):
        batch_rows = missing[offset:offset+config['batch_size']]
        images = []
        start = time.perf_counter()
        for row in batch_rows:
            with Image.open(image_path(project, config, row['image_id'])) as im:
                image = im.convert('RGB')
            images.append(ImageOps.mirror(image) if flip else image)
        decode_seconds = time.perf_counter()-start
        values, timings = backbone.encode(images)
        timings['image_decode_and_flip_seconds'] = decode_seconds
        for j, row in enumerate(batch_rows):
            start = time.perf_counter()
            arrays = {}
            for layer, d in DIMENSIONS[backbone.name].items():
                vector = values[layer][j].astype(np.float32)
                if vector.shape != (d,) or not np.isfinite(vector).all():
                    raise ValueError(f'{backbone.name}/{layer}: expected {d} finite components')
                arrays[layer] = vector
                arrays[layer + '_l2'] = l2(vector)
            image_timings = {k:v/len(batch_rows) for k,v in timings.items()}
            image_timings['normalization_seconds'] = time.perf_counter()-start
            npz(files[row['image_id']], image_id=np.array(row['image_id']), split=np.array(row['split']),
                metadata=np.array(__import__('json').dumps(dict(**backbone.metadata, batch_size=config['batch_size'],
                    timings_per_image=image_timings, flip=flip))), **arrays)
        print(f'{backbone.name}: extracted {min(offset+len(batch_rows), len(missing))}/{len(missing)} missing images', flush=True)
    save(output/'manifest.json', dict(identity=identity, metadata=backbone.metadata, flip=flip,
        files={i:str(p.relative_to(root)) for i,p in files.items()},
        file_sha256={str(p.relative_to(root)):file_hash(p) for p in files.values()}))
    representations = {}
    for layer in DIMENSIONS[backbone.name]:
        features, timings = {}, {}
        for image_id, path in files.items():
            with np.load(path, allow_pickle=False) as z:
                features[image_id] = z[layer + '_l2'].copy()
                timings[image_id] = __import__('json').loads(str(z['metadata']))['timings_per_image']
        name = backbone.name + '_' + layer
        representations[name] = dict(features=features, identity=identity, metadata=backbone.metadata,
            layer=layer, dimension=DIMENSIONS[backbone.name][layer], timings=timings,
            descriptor_bytes_per_image=4*DIMENSIONS[backbone.name][layer],
            raw_and_l2_bytes_per_image=8*DIMENSIONS[backbone.name][layer],
            extraction_cost_scope='shared forward pass across all outputs of this backbone',
            storage_bytes=sum(p.stat().st_size for p in files.values()))
    return representations
