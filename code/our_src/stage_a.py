"""Compatibility entry point for centroid training; evaluation lives in part1."""
from .part1.vocabulary import train_vocabularies, main
from .part1.data import manifest

if __name__ == '__main__':
    main()
