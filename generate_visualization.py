from vit_model.baselines.ViT.ViT_explanation_generator import LRP
from vit_model.VIT_LRP import vit_base_patch16_224_spectrogram as vit_LRP
from PIL import Image
from dataset.GTZAN import GTZAN
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization(original_image, attribution_generator, class_index=None, use_thresholding=False):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
      transformer_attribution = transformer_attribution * 255
      transformer_attribution = transformer_attribution.astype(np.uint8)
      ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      transformer_attribution[transformer_attribution == 255] = 1

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def print_top_classes(predictions, class_map, **kwargs):    
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []

    for cls_idx in class_indices:
        class_names.append(class_map[cls_idx])
        if len(class_map[cls_idx]) > max_str_len:
            max_str_len = len(class_map[cls_idx])
    
    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, class_map[cls_idx])
        output_string += ' ' * (max_str_len - len(class_map[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)

def swap_label_map(label_map):
    label_map_int = {}

    for key, value in label_map.items():
        label_map_int[value] = key

    return label_map_int

if __name__ == '__main__':
    dataset = GTZAN()
    LABEL_MAP = swap_label_map(dataset.label_map)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # initialize ViT pretrained
    model = vit_LRP(pretrained=True)
    attribution_generator = LRP(model)

    image = Image.open('samples/country00000.png').convert('RGB')
    transformed_image = transform(image)
    music_genre = generate_visualization(transformed_image, attribution_generator)


    print(type(transformed_image))
    print(type(music_genre))
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(spectr)
    # axs[0].axis('off')
    # axs[1].imshow(spectr)
    # axs[1].axis('off')

    plt.show()


    output = model(transformed_image.unsqueeze(0))
    print_top_classes(output, LABEL_MAP)


