import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
pdf_dir = '/doc-dataset'

MIME_MAPPING = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'bmp': 'image/bmp',
    'tif': 'image/tiff',
    'tiff': 'image/tiff',
    'webp': 'image/webp'
}
