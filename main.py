import time
start_time = time.time()
import numpy as np
import imageio
from skimage import filters, color
import matplotlib.pyplot as plt
class Seam_energy_minEnergy_pointer():
    def __init__(self, energy, x_coordinate_in_previous_row=None):
        self.energy = energy
        self.x_coordinate_in_previous_row = x_coordinate_in_previous_row

def min_seam_energy(pixel_energies, direction = 'vertical'):

    # Xoay hình ảnh 90 độ nếu hướng đường seam là ngang
    if direction == 'horizontal':
        pixel_energies = pixel_energies.T

    seam_energies = []
    # Khởi tạo năng lượng đường may ở hàng trên cùng
    #bằng cách sao chép năng lượng pixel ở hàng trên cùng.
    #Không có con trỏ quay lại ở hàng trên cùng.
    seam_energies.append([
        Seam_energy_minEnergy_pointer(pixel_energy)
        for pixel_energy in pixel_energies[0]
    ])
    # Bỏ qua hàng đầu tiên trong vòng lặp sau.

    for y in range(1, pixel_energies.shape[0]):
        pixel_energies_row = pixel_energies[y]
        seam_energies_row = []
        for x, pixel_energy in enumerate(pixel_energies_row):
            # Xác định phạm vi giá trị x cần lặp lại trong hàng trước đó.
            # Phạm vi phụ thuộc vào việc pixel hiện tại nằm ở giữa hình ảnh hay ở một trong các cạnh
            x_left = max(x - 1, 0)
            x_right = min(x + 1, len(pixel_energies_row) - 1)
            x_range = range(x_left, x_right + 1)
            min_parent_x = min(
                                x_range,
                                key=lambda x_i: seam_energies[y - 1][x_i].energy
            )
            min_seam_energy = Seam_energy_minEnergy_pointer(
                pixel_energy + seam_energies[y - 1][min_parent_x].energy,
                min_parent_x
            )
            seam_energies_row.append(min_seam_energy)
        seam_energies.append(seam_energies_row)

    # Đảo ngược hình ảnh trở lại nếu hướng đường may là ngang
    if direction == 'horizontal':
        seam_energies = np.array(seam_energies).T.tolist()

    return seam_energies

def retrieve_seam(seam_energies, direction = 'vertical'):

    # Tìm tọa độ x với năng lượng đường may nhỏ nhất ở hàng dưới cùng.
    if direction == 'vertical':
        # năng lượng đường seam là danh sách các mảng ở đây, tìm kiếm năng lượng tối thiểu ở hàng dưới cùng
        min_seam_end_x = min(
            range(len(seam_energies[-1])),
            key=lambda x: seam_energies[-1][x].energy
        )

        # Thực hiện theo các con trỏ phía sau để tạo danh sách các tọa độ tạo thành đường nối có năng lượng thấp nhất.
        seam = []
        seam_point_x = min_seam_end_x
        # Ngược lại, thêm tọa độ năng lượng tối thiểu cục bộ trở lại hàng trên cùng của hình ảnh
        for y in range(len(seam_energies) - 1, -1, -1):
            seam.append((seam_point_x, y))
            seam_point_x = \
                seam_energies[y][seam_point_x].x_coordinate_in_previous_row
        seam.reverse()

    elif direction == 'horizontal':
        # năng lượng seam tính bên dưới là một mảng không rõ ràng, được sử dụng cách tiếp cận khác với phương pháp trên,
        #  tìm kiếm năng lượng tối thiểu ở hầu hết cột bên phải
        seam_energies = np.array(seam_energies)
        min_seam_end_y = min(
            range(seam_energies.shape[0]),
            key=lambda x: seam_energies[:,-1][x].energy
        )

        # Thực hiện theo các con trỏ phía sau để tạo danh sách các tọa độ tạo thành đường nối có năng lượng thấp nhất.
        seam = []
        seam_point_y = min_seam_end_y
        # Ngược lại, thêm tọa độ năng lượng tối thiểu cục bộ trở lại cột bên trái hầu hết hình ảnh
        for x in range(seam_energies.shape[1] - 1, -1, -1):
            seam.append((x, seam_point_y))
            seam_point_y = \
                seam_energies[:, x][seam_point_y].x_coordinate_in_previous_row
        seam.reverse()

    return seam

def remove_seam(seam, img, direction = 'vertical'):

    if direction == 'vertical':
        # Tạo một bản sao của hình ảnh với kích thước giảm xuống
        new_pixels = np.zeros((img.shape[0], img.shape[1]-1, 3), dtype = 'float32')
        for row in range(img.shape[0]):
            offset = 0
            for col in range(img.shape[1]-1):
                if (col, row) in seam:
                    offset = 1
                new_pixels[row, col, :] = img[row, col + offset, :]

    if direction == 'horizontal':
        # Tạo một bản sao của hình ảnh với kích thước giảm xuống
        new_pixels = np.zeros((img.shape[0]-1, img.shape[1], 3), dtype = 'float32')
        for col in range(img.shape[1]):
            offset = 0
            for row in range(img.shape[0]-1):
                if (col, row) in seam:
                    offset = 1
                new_pixels[row, col, :] = img[row + offset, col, :]

    return new_pixels

def add_seam(seam, img, direction = 'vertical'):

    if direction == 'vertical':
        #Tạo một bản sao của hình ảnh với kích thước tăng lên
        new_pixels = np.zeros((img.shape[0], img.shape[1]+1, 3), dtype = 'float32')
        for row in range(img.shape[0]):
            offset = 0
            for col in range(img.shape[1]):
                if (col, row) in seam:
                    new_pixels[row, col, :] = img[row, col, :]
                    #copy pixel (cột, hàng) đến cột tiếp theo với giá trị pixel trung bình giữa các neighbors
                    new_pixels[row, col + 1, :] = (img[row, col, :] + img[row, col+1, :])/2
                    offset = 1
                    continue
                new_pixels[row, col+offset, :] = img[row, col, :]

    elif direction == 'horizontal':
        # Tạo một bản sao của hình ảnh với kích thước tăng lên
        new_pixels = np.zeros((img.shape[0] + 1, img.shape[1], 3), dtype='float32')
        for col in range(img.shape[1]):
            offset = 0
            for row in range(img.shape[0]):
                if (col, row) in seam:
                    new_pixels[row, col, :] = img[row, col, :]
                    # copy pixel (cột, hàng) sang hàng tiếp theo với giá trị pixel trung bình giữa các neighbors
                    new_pixels[row+1, col, :] = (img[row, col, :] + img[row+1, col, :])/2
                    offset = 1
                    continue
                new_pixels[row+offset, col, :] = img[row, col, :]

    return new_pixels

def fill_seam(seam, img):

    # thay đổi pixel hình ảnh trong đường nối thành các đường màu đỏ
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if (col, row) in seam:
                img[row, col, :] = [255,0,0]

    return img

def seam_mark(img, direction, pixels):

    # img là để tìm đường nối n (ở đây là pixel) đầu tiên cần loại bỏ, img_copy là để thêm đường nối
    img_copy = np.copy(img)
    lst_seam = []

    for i in range(pixels):
        print(f"Looking for {direction} seam No.{i + 1}")

        # Tính toán năng lượng img và gán cho mỗi điểm pixel
        pixel_energies = energy(img)

        # Tính toán list các mảng có năng lượng tối thiểu tại mỗi điểm pixel
        seam_energies = min_seam_energy(pixel_energies, direction)

        # Tìm đường nối đi từ trên xuống dưới hoặc từ trái sang phải
        seam = retrieve_seam(seam_energies, direction)

        # Thu thập tất cả các đường nối
        lst_seam.append(seam)

        # loại bỏ đường may và thay đổi img
        img = remove_seam(seam, img, direction)

    for i in range(pixels):
        print(f"Adding {direction} seam No.{i + 1}")
        seam = lst_seam.pop(0)

        # Điền vào đường nối trong img_copy với giá trị pixel màu đỏ
        img_copy = fill_seam(seam, img_copy)
        lst_seam = update_seams(lst_seam, seam, direction)

    return img_copy

def update_seams(remaining_seam, popped_seam, direction):

    updated_lst_seam = []

    if direction == 'vertical':
        for seam in remaining_seam:
            for i in range(len(popped_seam)):
                if popped_seam[i][0] < seam[i][0]:
                    seam[i] = list(seam[i])
                    seam[i][0] += 1
                    seam[i] = tuple(seam[i])
            updated_lst_seam.append(seam)

    elif direction == 'horizontal':
        for seam in remaining_seam:
            for i in range(len(popped_seam)):
                if popped_seam[i][1] < seam[i][1]:
                    seam[i] = list(seam[i])
                    seam[i][1] += 1
                    seam[i] = tuple(seam[i])
            updated_lst_seam.append(seam)

    return updated_lst_seam

def energy(img):

    # dùng hàm Sobel cho gradient năng lượng tại mỗi điểm pixel trong img
    img_gradient = filters.sobel(color.rgb2gray(img))

    return img_gradient

def seam_carving_reducing(img,direction,pixels):

    # Lặp lại loại bỏ đường may trong mỗi vòng lặp
    for i in range(pixels):
        print(f"Removing {direction} seam No.{i+1}")

        # Tính toán năng lượng img và gán cho mỗi điểm pixel
        pixel_energies = energy(img)

        # Tính toán danh sách các mảng có năng lượng tối thiểu tại mỗi điểm pixel
        seam_energies = min_seam_energy(pixel_energies, direction)

        # Tìm đường nối đi từ trên xuống dưới hoặc từ trái sang phải
        seam = retrieve_seam(seam_energies, direction)

        #return hình ảnh mới với đường nối đã bị loại bỏ
        img = remove_seam(seam, img, direction)

    return img

def seam_carving_increasing(img,direction,pixels):

    # img là để tìm đường nối n (ở đây là pixel) đầu tiên cần loại bỏ, img_copy là để thêm đường nối
    img_copy = np.copy(img)
    lst_seam = []

    for i in range(pixels):
        print(f"Looking for {direction} seam No.{i + 1}")

        # Tính toán năng lượng img và gán cho mỗi điểm pixel
        pixel_energies = energy(img)

        # Tính toán danh sách các mảng có năng lượng tối thiểu tại mỗi điểm pixel
        seam_energies = min_seam_energy(pixel_energies, direction)

        # Tìm đường nối đi từ trên xuống dưới hoặc từ trái sang phải
        seam = retrieve_seam(seam_energies, direction)

        # Thu thập tất cả các đường nối
        lst_seam.append(seam)

        # Loại bỏ đường may và thay đổi img
        img = remove_seam(seam, img, direction)

    for i in range(pixels):
        print(f"Adding {direction} seam No.{i + 1}")
        seam = lst_seam.pop(0)

        # Thêm các đường nối để xóa chúng, sau đó cập nhật tọa độ của chúng nếu cần
        img_copy = add_seam(seam, img_copy, direction)
        lst_seam = update_seams(lst_seam, seam, direction)

    return img_copy

if __name__ == '__main__':

    print("\nSEAM CARVING\n" )
    print('*-' * 10 + " MENU " + '*-' * 10)
    print()

    # INPUT
    pic = input("Chọn ảnh (Ex: abc.jpg):")
    operation = input("Tăng kích thước/ giảm kích thước (Ex: increase/decrease):")
    pixels = input("Số pixel để thực hiện:")
    direction = input("Đường seam theo chiều: (Ex: height/width):")
    mark = input("In ra các đường seam (y/n):")
    print()

    # read the input image
    img = imageio.imread(pic).astype(np.float32)

    # Xác định hướng đường may từ hướng
    h = 'horizontal'
    v = 'vertical'
    if direction == 'height':
        seam_direction = h
    elif direction == 'width':
        seam_direction = v

    # Determine which operation function to use
    if operation == 'increase':
        img_resized = seam_carving_increasing(img, seam_direction, int(pixels))
    elif operation == 'decrease':
        img_resized = seam_carving_reducing(img, seam_direction, int(pixels))

    # Display user inputs
    print(f"Thông số: {pic}, Mode: {operation} chiều: {direction}, số pixels: {pixels}")
    print(f"Ảnh gốc: {img.shape[0]}x{img.shape[1]}")
    print(f"Ảnh sau khi resize: {img_resized.shape[0]}x{img_resized.shape[1]}")

    # Display original image and resized image
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1,2,1)
    original_img_plot = plt.imshow(img.astype(np.uint8))
    plt.title('Ảnh gốc')
    fig.add_subplot(1,2,2)
    resized_img_plot = plt.imshow(img_resized.astype(np.uint8))
    plt.title('Ảnh sau khi thay đổi')
    plt.show()

    # Save resized image in default pattern name
    img_name = pic.split(".")[0]
    output_name_tuple = (operation, img_name, direction, pixels, '.jpg')
    output_name = "_".join(output_name_tuple)
    img_resized = img_resized.astype(np.uint8)
    imageio.imwrite(output_name, img_resized)
    print(f"Lưu ảnh {output_name}")
    print()

    #Nếu cần in ra các đường seam của hình ảnh
    if mark == 'y':
        img_mark = seam_mark(img, seam_direction, int(pixels))

        img_mark_name = 'seam_' + img_name +  '_' + seam_direction + '_' + pixels + '.jpg'
        img_mark = img_mark.astype(np.uint8)
        imageio.imwrite(img_mark_name, img_mark)

        plt.figure()
        plt.title('Hiển thị các đường seam')
        plt.imshow(img_mark)
        plt.show()



end_time = time.time()
elapsed_time = end_time - start_time
min= elapsed_time //60
sec= elapsed_time -60*min
print ("Thời gian thực hiện: " + str(min)+ " min " + str (sec) + " sec")





