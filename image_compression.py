import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def compress_image(image_path):
    image = Image.open(image_path).convert('L')
    print('Image size:', image.size)
    col, row = image.size
    original_image_size = row * col 

    # PIL -> numpy.array
    A = np.array(image, 'uint8')

    # SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print('The shape of U, S, Vt:', U.shape, S.shape, Vt.shape)

    ranks = [4, 10, 20, 50, 128]
    plt.figure(figsize=(18, 10))

    for i, rank in enumerate(ranks, 1):
        # The @ operator can be used as a shorthand for np.matmul on ndarrays.
        new_image = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        compressed_image_size = rank * (row + col + 1)
        ratio = compressed_image_size / original_image_size * 100
    
        plt.subplot(1, len(ranks), i)
        plt.imshow(new_image, cmap='gray')
        plt.title(f'Rank: {rank}, Ratio: {ratio:.2f}')
    
    new_image, _ = image_path.split('.')
    new_image_path = f'{new_image}_compressed.png'
    plt.savefig(f'{new_image_path}')
    print(f'{new_image_path} saved successfully.\n')


if __name__ == '__main__':
    compress_image('baboon.png')
    compress_image('runner1.jpg')
