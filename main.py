from .main_core_processing import *


def processing():

    print('Handling the images...\n')

    print("---------------------- Image 1 ---------------------\n")

    image_1 = cv2.imread(r'Images\baby.tif')
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_1_clean = clean_image_1(image_1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_1_clean, cmap='gray', vmin=0, vmax=255)

    print("---------------------- Image 2 ---------------------\n")
    image_2 = cv2.imread(r'Images\windmill.tif')
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    image_2_clean = clean_image_2(image_2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2_clean, cmap='gray', vmin=0, vmax=255)

    print("---------------------- Image 3 ---------------------\n")
    image_3 = cv2.imread(r'Images\watermelon.tif')
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    image_3_clean = clean_image_3(image_3)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_3, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_3_clean, cmap='gray', vmin=0, vmax=255)

    print("---------------------- Image 4 ---------------------\n")
    image_4 = cv2.imread(r'Images\umbrella.tif')
    image_4 = cv2.cvtColor(image_4, cv2.COLOR_BGR2GRAY)
    image_4_clean = clean_image_4(image_4)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_4, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_4_clean, cmap='gray', vmin=0, vmax=255)

    print("---------------------- Image 5 ---------------------\n")
    image_5 = cv2.imread(r'Images\USAflag.tif')
    image_5 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)
    image_5_clean = clean_image_5(image_5)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_5, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_5_clean, cmap='gray', vmin=0, vmax=255)

    print("---------------------- Image 6 ---------------------\n")
    image_6 = cv2.imread(r'Images\cups.tif')
    image_6 = cv2.cvtColor(image_6, cv2.COLOR_BGR2GRAY)
    image_6_clean = clean_image_6(image_6)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_6, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_6_clean, cmap='gray', vmin=0, vmax=255)

    print("---------------------- Image 7 ---------------------\n")
    image_7 = cv2.imread(r'Images\house.tif')
    image_7 = cv2.cvtColor(image_7, cv2.COLOR_BGR2GRAY)
    image_7_clean = clean_image_7(image_7)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_7, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_7_clean, cmap='gray')

    print("---------------------- Image 8 ---------------------\n")
    image_8 = cv2.imread(r'Images\bears.tif')
    image_8 = cv2.cvtColor(image_8, cv2.COLOR_BGR2GRAY)
    image_8_clean = clean_image_8(image_8)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_8, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(image_8_clean, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == "__main__":
    processing()
