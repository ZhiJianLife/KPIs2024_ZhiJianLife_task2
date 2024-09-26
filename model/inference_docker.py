from mmseg.apis import init_segmentor, inference_segmentor
import glob
import os
import logging
from PIL import Image
import numpy as np
import openslide
from tqdm import tqdm
import imageio
import cv2
import re
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
Image.MAX_IMAGE_PIXELS = None





#  组织区域
def post_process(ori_img_path):
    img_mask = Image.open(ori_img_path).convert('L')
    img_mask = np.array(img_mask)
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            cv2.drawContours(img_mask, [contour], -1, 0, thickness=cv2.FILLED)
    if ori_img_path.split('/')[-2] != "NEP25":
        img_mask = cv2.bitwise_not(img_mask)
    cv2.imwrite(ori_img_path, img_mask)
    np.save(ori_img_path.replace('.png', '.npy'), np.array(img_mask))
def run_post_process(args):
    print("开始预处理：")
    logging.basicConfig(level=logging.INFO)
    ff = os.walk( args.wsi_path )
    paths = []
    for root, dirs, files in ff:
        for file in files:
            if file.split('_')[-1] == 'wsi.tiff':
                paths.append( os.path.join( root, file ) )
    for path in paths:
        print(path)
        print(path.split('/')[-3])
        npy_name = os.path.basename(path)
        npy_path = os.path.join(args.npy_path, npy_name[:-4] + 'npy')
        slide = openslide.OpenSlide(path)
        img_RGB = np.array(slide.read_region((0, 0),
                               args.level,
                               slide.level_dimensions[args.level]).convert('RGB'))
        slide.close()
        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_mask = tissue_RGB ##& tissue_S & min_R & min_G & min_B
        np.save(npy_path, tissue_mask)
        creat_path = os.path.join(args.gray_path, path.split('/')[-3])
        os.makedirs(creat_path, exist_ok=True)
        gray_path = os.path.join(args.gray_path, path.split('/')[-3],npy_name[:-4] + 'png')

        print(gray_path)
        im = Image.fromarray(tissue_mask)
        if im.mode == "F":
            im = im.convert('RGB')
        im.save(gray_path)
        if args.post_process:
            post_process(gray_path)
    print("预处理结束")





# #切patch
def wsi_to_patch(args):
    print("开始将wsi切patch")
    subfolders = [f.path for f in os.scandir(args.wsi_path) if f.is_dir()]
    subfolders_notnor = []
    subfolders_nor = []
    subfolders_NEP = []
    for i in subfolders:
        if i.endswith('normal'):
            subfolders_nor.append(i)
        elif i.endswith('NEP25'):
            subfolders_NEP.append(i)
        else:
            subfolders_notnor.append(i)
    for folder in subfolders_nor:
        paths = glob.glob(os.path.join(folder,'img','*_wsi.tiff'))
        for idx_svs, path in enumerate(paths):
            if args.select_txt_path is not None:
                with open(args.select_txt_path, 'r') as f:
                    label_list = f.read().splitlines()
                    if os.path.basename(path) not in label_list:
                        continue
            print('{}/{} file {} is processing'.format(str(idx_svs + 1), str(len(paths)), path))
            svs_name = os.path.basename(path)[:-4]
            slide = openslide.OpenSlide(path)
            slide_x, slide_y = slide.level_dimensions[0]
            print(slide_x,slide_y)
            print(os.path.basename(path)[:-4])
            # 加载mask文件
            mask_path = os.path.join(args.mask_path, path.split("/")[-3], os.path.basename(path)[:-4] + 'npy')
            print(mask_path)
            mask = np.load(mask_path)
            patch_size = 4096
            step = 2048 # 不重叠的步长等于patch大小
            coords = set()
            print(mask.shape[0], mask.shape[1])
            for x in range(0, mask.shape[1], step):
                for y in range(0, mask.shape[0], step):
                    patch = mask[y:y + patch_size, x:x + patch_size]
                    if patch.mean() > 0:
                        coords.add((x, y))
            slide_rect_y, slide_rect_x = mask.shape
            print(mask.shape[0], mask.shape[1])
            for coord in tqdm(coords):
                cord_x = int(coord[0])
                cord_y = int(coord[1])
                # 计算图像块的坐标
                start_x = min(cord_x, slide_rect_x - patch_size)
                start_y = min(cord_y, slide_rect_y - patch_size)
                # 提取图像块并转换为RGB
                img = slide.read_region((start_x, start_y), 0, (patch_size, patch_size)).convert('RGB')
                img_path = os.path.join(args.patch_path, path.split('/')[-3],svs_name.split('_wsi')[0], 'image', svs_name + '_'
                                        + str(start_y) + '_' + str(start_x) + '.png')
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                print("img_patch",img_path)
                print("--------------------------")
                img.save(img_path)
            slide.close()

    for folder in subfolders_NEP:
        paths = glob.glob(os.path.join(folder, 'img','*_wsi.tiff'))
        for idx_svs, path in enumerate(paths):
            if args.select_txt_path is not None:
                with open(args.select_txt_path, 'r') as f:
                    label_list = f.read().splitlines()
                    if os.path.basename(path) not in label_list:
                        continue
            print('{}/{} file {} is processing'.format(str(idx_svs + 1), str(len(paths)), path))
            svs_name = os.path.basename(path)[:-4]
            slide = openslide.OpenSlide(path)
            slide_rect_x, slide_rect_y = slide.level_dimensions[0]
            print(slide_rect_x,slide_rect_y)
            # 加载mask文件
            mask_path = os.path.join(args.mask_path, path.split("/")[-3], os.path.basename(path)[:-4] + 'npy')
            mask = np.load(mask_path)
            patch_size_NEP_nor = 2048
            step = 1024
            coords = set()
            print(mask.shape[0], mask.shape[1])
            for x in range(0, mask.shape[0], step):
                for y in range(0, mask.shape[1], step):
                    patch = mask[x:x + patch_size_NEP_nor, y:y + patch_size_NEP_nor]
                    if patch.mean() > 0:
                        coords.add((x, y))
            print(len(coords))

            for coord in tqdm(coords):
                cord_x = int(coord[0])
                cord_y = int(coord[1])
                print(f'Extracting patch at: ({cord_x}, {cord_y})')
                # 提取图像块并转换为RGB
                img = slide.read_region((cord_y, cord_x), 0, (patch_size_NEP_nor, patch_size_NEP_nor)).convert('RGB')
                start_x = max(cord_x, 0)
                start_y = max(cord_y, 0)
                img_path = os.path.join(args.patch_path, path.split('/')[-3],svs_name.split('_wsi')[0], 'image', svs_name + '_'
                                        + str(start_y) + '_' + str(start_x) + '.png')
                os.makedirs(os.path.dirname(img_path), exist_ok=True)

                # 保存图像块和肿瘤mask块
                print("img_patch",img_path)
                print("--------------------------")
                # 保存图像块和肿瘤mask块
                img.save(img_path)
            slide.close()

    for folder in subfolders_notnor:
        paths = glob.glob(os.path.join(folder,'img','*_wsi.tiff'))
        for idx_svs, path in enumerate(paths):
            if args.select_txt_path is not None:
                with open(args.select_txt_path, 'r') as f:
                    label_list = f.read().splitlines()
                    if os.path.basename(path) not in label_list:
                        continue
            print('{}/{} file {} is processing'.format(str(idx_svs + 1), str(len(paths)), path))
            svs_name = os.path.basename(path)[:-4]
            slide = openslide.OpenSlide(path)
            slide_rect_x, slide_rect_y = slide.level_dimensions[0]
            print(slide_rect_x,slide_rect_y)
            # 加载mask文件
            mask_path = os.path.join(args.mask_path, path.split("/")[-3], os.path.basename(path)[:-4] + 'npy')
            mask = np.load(mask_path)
            patch_size_NEP_nor = 4096
            step = 2048
            coords = set()
            print(mask.shape[0], mask.shape[1])
            for x in range(0, mask.shape[0], step):
                for y in range(0, mask.shape[1], step):
                    patch = mask[x:x + patch_size_NEP_nor, y:y + patch_size_NEP_nor]
                    if patch.mean() > 0:
                        coords.add((x, y))
            for coord in tqdm(coords):
                cord_x = int(coord[0])
                cord_y = int(coord[1])
                print(f'Extracting patch at: ({cord_x}, {cord_y})')
                # 提取图像块并转换为RGB
                img = slide.read_region((cord_y, cord_x), 0, (patch_size_NEP_nor, patch_size_NEP_nor)).convert('RGB')
                start_x = max(cord_x, 0)
                start_y = max(cord_y, 0)
                img_path = os.path.join(args.patch_path, path.split('/')[-3],svs_name.split('_wsi')[0], 'image', svs_name + '_'
                                        + str(start_y) + '_' + str(start_x) + '.png')
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                print("img_patch",img_path)
                print("--------------------------")
                img.save(img_path)
            slide.close()
    print("切patch结束")

def inference(args):
    print("开始推理patch")
    model = init_segmentor(args.config_file, args.checkpoint_file, device='cuda:0')
    ff = os.walk(args.test_img_dir)
    img_list = []
    for root, dirs, files in ff:
        if os.path.basename(root) == 'image':
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    img_list.append(os.path.join(root, file))
    for i, img in enumerate(img_list):
        print('Infenencing image {}/{}, image name is {}'.format(str(i), str(len(img_list)), img))
        result = inference_segmentor(model, img)
        save_dir_new = args.save_dir + img.split('/')[-4] + '/' + img.split('/')[-3] + '/'
        result = result[0] * 255
        os.makedirs(os.path.dirname(save_dir_new), exist_ok=True)
        cv2.imwrite(os.path.join(save_dir_new, os.path.basename(img).replace('img', 'mask')), result)
    print("推理patch结束")

#聚合代码
def patch_to_wsi(args):
    print("开始聚合patch")
    # 获取所有patch文件路径
    for root, dirs, files in os.walk(args.save_dir):
        for dir in dirs:
            if dir == 'normal':
                subdir = os.path.join(root, dir)
                # 遍历每个子文件夹中的图像文件
                for file in os.listdir(subdir):
                    coords = []
                    file_path = os.path.join(subdir, file)
                    for f in os.listdir(file_path):
                        match = re.search(r'_(\d+)_(\d+)\.png', os.path.basename(f))
                        if match:
                            coords.append((file_path+'/'+f, int(match.group(1)), int(match.group(2))))
                    wsi_path = os.path.join(args.wsi_path, dir, 'img',file +"_wsi.tiff")
                    print(wsi_path)
                    slide = openslide.OpenSlide(wsi_path)
                    slide_rect_x, slide_rect_y = slide.level_dimensions[0]
                    # 如果有mask文件，需要类似的处理
                    reassembled_mask = Image.new('L', (slide_rect_x, slide_rect_y))
                    x_coords = [coord[1] for coord in coords]
                    y_coords = [coord[2] for coord in coords]
                    max_x = max(x_coords)
                    min_x = min(x_coords)
                    max_y = max(y_coords)
                    min_y = min(y_coords)
                    for (file, coord_x, coord_y) in coords:
                        mask_patch = Image.open(file)
                        if coord_y == min_y or coord_y == max_y or coord_x == min_x or coord_x == max_x:
                            reassembled_mask.paste(mask_patch, (coord_y, coord_x))
                        else:
                            cropped_patch = mask_patch.crop((1024, 1024, 3072, 3072))
                            reassembled_mask.paste(cropped_patch, (coord_y + 1024, coord_x + 1024))
                    reassembled_mask = ndi.zoom(reassembled_mask, (1/2, 1/2), order=1)
                    reassembled_mask = Image.fromarray(np.uint8(reassembled_mask))
                    reassembled_mask_tiff_path = os.path.join(args.output_path, dir, f.split(".")[0].replace("wsi", "mask") + ".tiff")
                    reassembled_mask.save(reassembled_mask_tiff_path)

            elif dir == 'NEP25':
                subdir = os.path.join(root, dir)
                # 遍历每个子文件夹中的图像文件
                for file in os.listdir(subdir):
                    coords = []
                    file_path = os.path.join(subdir, file)
                    for f in os.listdir(file_path):
                        match = re.search(r'_(\d+)_(\d+)\.png', os.path.basename(f))
                        if match:
                            coords.append((file_path+'/'+f, int(match.group(1)), int(match.group(2))))
                    wsi_path = os.path.join(args.wsi_path, dir, 'img',file+"_wsi.tiff")
                    print(wsi_path)
                    slide = openslide.OpenSlide(wsi_path)
                    slide_rect_x, slide_rect_y = slide.level_dimensions[0]
                    print(slide_rect_x, slide_rect_y)
                    # 如果有mask文件，需要类似的处理
                    reassembled_mask = Image.new('L', (slide_rect_x, slide_rect_y))
                    x_coords = [coord[1] for coord in coords]
                    y_coords = [coord[2] for coord in coords]
                    max_x = max(x_coords)
                    min_x = min(x_coords)
                    max_y = max(y_coords)
                    min_y = min(y_coords)
                    # #normal是x,y，其他的是y,x
                    for (file, coord_x, coord_y) in coords:
                        mask_patch = Image.open(file)
                        if coord_y == min_y or coord_y == max_y or coord_x == min_x or coord_x == max_x:
                            reassembled_mask.paste(mask_patch, (coord_x, coord_y))
                        else:
                            cropped_patch = mask_patch.crop((512, 512, 1536, 1536))
                            reassembled_mask.paste(cropped_patch, (coord_x + 512, coord_y + 512))
                    reassembled_mask_tiff_path = os.path.join(args.output_path, dir, f.split(".")[0].replace("wsi", "mask") + ".tiff")
                    reassembled_mask.save(reassembled_mask_tiff_path)


            elif dir == '56Nx' or dir == 'DN':
                subdir = os.path.join(root, dir)
                # 遍历每个子文件夹中的图像文件
                for file in os.listdir(subdir):
                    coords = []
                    file_path = os.path.join(subdir, file)
                    for f in os.listdir(file_path):
                        match = re.search(r'_(\d+)_(\d+)\.png', os.path.basename(f))
                        if match:
                            coords.append((file_path+'/'+f, int(match.group(1)), int(match.group(2))))
                    wsi_path = os.path.join(args.wsi_path, dir, 'img',file+"_wsi.tiff")
                    print(wsi_path)
                    slide = openslide.OpenSlide(wsi_path)
                    slide_rect_x, slide_rect_y = slide.level_dimensions[0]
                    # 如果有mask文件，需要类似的处理
                    reassembled_mask = Image.new('L', (slide_rect_x, slide_rect_y))
                    x_coords = [coord[1] for coord in coords]
                    y_coords = [coord[2] for coord in coords]
                    max_x = max(x_coords)
                    min_x = min(x_coords)
                    max_y = max(y_coords)
                    min_y = min(y_coords)
                    # #normal是x,y，其他的是y,x
                    for (file, coord_x, coord_y) in coords:
                        mask_patch = Image.open(file)
                        if coord_y == min_y or coord_y == max_y or coord_x == min_x or coord_x == max_x:
                            reassembled_mask.paste(mask_patch, (coord_x, coord_y))
                        else:
                            cropped_patch = mask_patch.crop((1024, 1024, 3072, 3072))
                            reassembled_mask.paste(cropped_patch, (coord_x + 1024, coord_y + 1024))
                    reassembled_mask = ndi.zoom(reassembled_mask,(1/2, 1/2),order=1)
                    reassembled_mask = Image.fromarray(np.uint8(reassembled_mask))
                    reassembled_mask_tiff_path = os.path.join(args.output_path, dir, f.split(".")[0].replace("wsi", "mask") + ".tiff")
                    reassembled_mask.save(reassembled_mask_tiff_path)
            else:
                break
    print("聚合patch结束")
    print("看到这里，说明这个docker推理成功啦。1、请原谅我们没有使用官方的预处理方法，自己编写了预处理代码，包括获得组织区域、crop patch、推理、聚合。2、结果保存在/output/目录下，如果有任何问题，请通过电子邮箱联系我们，谢谢您，祝您生活愉快！")



# 示例args
class Args:
    level = 0
    post_process = 1
    wsi_path = '../input/'
    npy_path = '../process/'
    gray_path = '../process/ori/'

    mask_path = '../process/ori/'
    tumor_path = '../input/'
    patch_path = '../process/patch/'
    select_txt_path = None

    config_file = '../model/tools/configs/upernet/upernet_r101_769x769_80k_cityscapes.py'
    checkpoint_file = '../model/checkpoint/best_model.pth'
    test_img_dir = '../process/patch/'
    save_dir = '../process/inference_result/'
    patch_size = 4096
    output_path = '../output/'


args = Args()
run_post_process(args)
wsi_to_patch(args)
inference(args)
patch_to_wsi(args)

















