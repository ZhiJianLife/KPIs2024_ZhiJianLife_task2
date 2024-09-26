#40x放大倍率做patch
import os
import glob
import numpy as np
import openslide
from PIL import Image
import imageio
from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"]='1'
Image.MAX_IMAGE_PIXELS = None


def run(args):
    subfolders = [f.path for f in os.scandir(args.wsi_path) if f.is_dir()]
    subfolders_nor = subfolders[-1]
    subfolders_notnorNEP = [subfolders[0],subfolders[2]]
    subfolders_NEP = subfolders[1]
    for folder in subfolders_nor:
        paths = glob.glob(os.path.join(folder, '*_wsi.tiff'))
        for idx_svs, path in enumerate(paths):
            if args.select_txt_path is not None:
                with open(args.select_txt_path, 'r') as f:
                    label_list = f.read().splitlines()
                    if os.path.basename(path) not in label_list:
                        continue
            print('{}/{} file {} is processing'.format(str(idx_svs + 1), str(len(paths)), path))
            svs_name = os.path.basename(path)[:-4]
            slide = openslide.OpenSlide(path)
            print(os.path.basename(path)[:-4])
            # 加载mask文件
            mask_path = os.path.join(args.mask_path, path.split('/')[-2],os.path.basename(path)[:-4] + 'npy')
            print(mask_path)
            mask = np.load(mask_path)
            patch_size = 2048
            step = 1024  # 不重叠的步长等于patch大小

            coords = set()
            print(mask.shape[1])
            for x in range(0, mask.shape[1], step):
                for y in range(0, mask.shape[0], step):
                    patch = mask[x:x + patch_size, y:y + patch_size]
                    if patch.mean()>0:
                        print(patch.mean(),x,y)
                        coords.add((x, y))

            # 加载肿瘤mask文件
            tumor_mask_path = os.path.join(args.tumor_path,path.split('/')[-2], os.path.basename(path)[:-4].split('_wsi')[0] + '_mask.tiff')
            tumor_mask = Image.open(tumor_mask_path)
            tumor_mask = np.array(tumor_mask)
            slide_rect_y, slide_rect_x = tumor_mask.shape
            for coord in tqdm(coords):
                cord_x = int(coord[0])
                cord_y = int(coord[1])

                # 计算图像块的坐标
                start_x = min(cord_x, slide_rect_x - patch_size)
                start_y = min(cord_y, slide_rect_y - patch_size)

                # 提取图像块并转换为RGB
                img = slide.read_region((start_x, start_y), 0, (patch_size, patch_size)).convert('RGB')

                # 提取对应的肿瘤mask图像块
                img_label = tumor_mask[start_y:start_y + patch_size, start_x:start_x + patch_size]

                # 生成图像块和mask块的保存路径
                img_path = os.path.join(args.patch_path, path.split('/')[-2],svs_name.split('_wsi')[0], 'image', svs_name + '_'
                                        + str(start_y) + '_' + str(start_x) + '.png')
                label_path = os.path.join(args.patch_path, path.split('/')[-2],svs_name.split('_wsi')[0],'mask', svs_name + '_'
                                          + str(start_y) + '_' + str(start_x) + '.png')
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                # 保存图像块和肿瘤mask块
                print("img_patch",img_path)
                print("label_patch", label_path)
                print("--------------------------")
                img.save(img_path)
                imageio.imwrite(label_path, img_label)
            slide.close()

    for folder in subfolders_NEP:
        paths = glob.glob(os.path.join(folder, '*_wsi.tiff'))
        for idx_svs, path in enumerate(paths):
            if args.select_txt_path is not None:
                with open(args.select_txt_path, 'r') as f:
                    label_list = f.read().splitlines()
                    if os.path.basename(path) not in label_list:
                        continue

            print('{}/{} file {} is processing'.format(str(idx_svs + 1), str(len(paths)), path))
            svs_name = os.path.basename(path)[:-4]
            slide = openslide.OpenSlide(path)
            # 加载mask文件
            mask_path = os.path.join(args.mask_path, os.path.basename(path)[:-4] + 'npy')
            mask = np.load(mask_path)
            patch_size_NEP = 4096
            step = 2048
            coords = set()
            for x in range(0, mask.shape[0], step):
                for y in range(0, mask.shape[1], step):
                    patch = mask[x:x + patch_size_NEP, y:y + patch_size_NEP]
                    if patch.mean() > 0:
                        coords.add((x, y))

            # 加载肿瘤mask文件
            tumor_mask_path = os.path.join(args.tumor_path, path.split("/")[-2],os.path.basename(path)[:-4].split('_wsi')[0] + '_mask.tiff')
            tumor_mask = Image.open(tumor_mask_path)
            tumor_mask = np.array(tumor_mask)
            slide_rect_x, slide_rect_y = tumor_mask.shape
            print("nihao", tumor_mask.shape)
            print(len(coords))

            for coord in tqdm(coords):
                cord_x = int(coord[0])
                cord_y = int(coord[1])
                print(f'Extracting patch at: ({cord_x}, {cord_y})')
                # 提取图像块并转换为RGB
                img = slide.read_region((cord_x, cord_y), 0, (patch_size_NEP, patch_size_NEP)).convert('RGB')
                start_x = max(cord_x, 0)
                end_x = min(cord_x + patch_size_NEP, slide_rect_x)

                print(cord_x + patch_size_NEP, slide_rect_x, end_x)
                start_y = max(cord_y, 0)
                end_y = min(cord_y + patch_size_NEP, slide_rect_y)

                print(cord_y+patch_size_NEP,slide_rect_y,end_y)
                # 提取对应的肿瘤mask图像块
                img_label = tumor_mask[start_x:end_x, start_y:end_y]
                img_label = Image.fromarray(img_label)
                # 生成图像块和mask块的保存路径
                img_path = os.path.join(args.patch_path, path.split('/')[-2],svs_name.split('_wsi')[0], 'image', svs_name + '_'
                                        + str(start_x) + '_' + str(start_y) + '.png')
                label_path = os.path.join(args.patch_path, path.split('/')[-2],svs_name.split('_wsi')[0],'mask', svs_name + '_'
                                          + str(start_x) + '_' + str(start_y) + '.png')
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                # 保存图像块和肿瘤mask块
                print("img_patch",img_path)
                print("label_patch", label_path)
                print("--------------------------")
                # 保存图像块和肿瘤mask块
                img.save(img_path)
                img_label.save(label_path)
            slide.close()

    for folder in subfolders_notnorNEP:
        paths = glob.glob(os.path.join(folder, '*_wsi.tiff'))
        for idx_svs, path in enumerate(paths):
            if args.select_txt_path is not None:
                with open(args.select_txt_path, 'r') as f:
                    label_list = f.read().splitlines()
                    if os.path.basename(path) not in label_list:
                        continue
            print('{}/{} file {} is processing'.format(str(idx_svs + 1), str(len(paths)), path))
            svs_name = os.path.basename(path)[:-4]
            slide = openslide.OpenSlide(path)
            # 加载mask文件
            mask_path = os.path.join(args.mask_path, os.path.basename(path)[:-4] + 'npy')
            mask = np.load(mask_path)
            patch_size_NEP_nor = 2048
            step = 1024
            coords = set()
            for x in range(0, mask.shape[0], step):
                for y in range(0, mask.shape[1], step):
                    patch = mask[x:x + patch_size_NEP_nor, y:y + patch_size_NEP_nor]
                    if patch.mean() > 0:
                        coords.add((x, y))

            # 加载肿瘤mask文件
            tumor_mask_path = os.path.join(args.tumor_path, path.split("/")[-2],os.path.basename(path)[:-4].split('_wsi')[0] + '_mask.tiff')
            tumor_mask = Image.open(tumor_mask_path)
            tumor_mask = np.array(tumor_mask)
            slide_rect_x, slide_rect_y = tumor_mask.shape
            print("nihao", tumor_mask.shape)
            print(len(coords))

            for coord in tqdm(coords):
                cord_x = int(coord[0])
                cord_y = int(coord[1])
                print(f'Extracting patch at: ({cord_x}, {cord_y})')
                # 提取图像块并转换为RGB
                img = slide.read_region((cord_x, cord_y), 0, (patch_size_NEP_nor, patch_size_NEP_nor)).convert('RGB')
                start_x = max(cord_x, 0)
                end_x = min(cord_x + patch_size_NEP_nor, slide_rect_x)

                print(cord_x + patch_size_NEP_nor, slide_rect_x, end_x)
                start_y = max(cord_y, 0)
                end_y = min(cord_y + patch_size_NEP_nor, slide_rect_y)

                print(cord_y+patch_size_NEP_nor,slide_rect_y,end_y)
                # 提取对应的肿瘤mask图像块
                img_label = tumor_mask[start_x:end_x, start_y:end_y]
                img_label = Image.fromarray(img_label)
                # 生成图像块和mask块的保存路径
                img_path = os.path.join(args.patch_path, path.split('/')[-2],svs_name.split('_wsi')[0], 'image', svs_name + '_'
                                        + str(start_x) + '_' + str(start_y) + '.png')
                label_path = os.path.join(args.patch_path, path.split('/')[-2],svs_name.split('_wsi')[0],'mask', svs_name + '_'
                                          + str(start_x) + '_' + str(start_y) + '.png')
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                # 保存图像块和肿瘤mask块
                print("img_patch",img_path)
                print("label_patch", label_path)
                print("--------------------------")
                # 保存图像块和肿瘤mask块
                img.save(img_path)
                img_label.save(label_path)
            slide.close()



# 示例args
class Args:
    wsi_path = '/home/data/KPIs/data/task2_train/val/'
    mask_path = '/home/data/KPIs/data/task2_train/tissue/'
    tumor_path = '/home/data/KPIs/data/task2_train/val/'
    patch_path = '/home/data/KPIs/data/task2_train/patch/'
    select_txt_path = None
args = Args()
run(args)

