import os

def check_images_in_captions(image_dir, captions_file):
    """
    Kiểm tra xem tất cả các tên tệp ảnh trong một thư mục có tồn tại trong tệp chú thích hay không.

    Args:
        image_dir (str): Đường dẫn đến thư mục chứa ảnh.
        captions_file (str): Đường dẫn đến tệp chú thích.
    """
    try:
        # 1. Lấy tất cả tên tệp ảnh từ thư mục
        image_files = {f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))}
        if not image_files:
            print(f"Cannot find any image in {image_dir}")
            return

        # 2. Đọc nội dung của tệp chú thích
        print(f"Reading content from {captions_file}...")
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions_content = f.read()
        print("Done reading captions file.")

        # 3. Tìm những tệp ảnh không được đề cập trong tệp chú thích
        missing_images = []
        print("Start checking images...")
        for image_file in image_files:
            if image_file not in captions_content:
                missing_images.append(image_file)

        # 4. Báo cáo kết quả
        if not missing_images:
            print("\nAll images in the directory are in the captions file.")
        else:
            print(f"\nFound {len(missing_images)} images that are not in the captions file:")
            for img in missing_images:
                print(f"- {img}")

    except FileNotFoundError as e:
        print(f"Error: Cannot find file or directory. Please check the path.")
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    IMAGE_DIR = 'data/images'
    CAPTIONS_FILE = 'data/captions.txt'
    check_images_in_captions(IMAGE_DIR, CAPTIONS_FILE) 