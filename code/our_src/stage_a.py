"""Compatibility entry point for centroid training; evaluation lives in stagea."""
from .stagea.vocabulary import train_vocabularies, main
from .stagea.data import manifest

if __name__ == '__main__':
    main()
