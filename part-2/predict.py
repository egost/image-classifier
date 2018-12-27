###########
# Imports #
###########
# Core
import os
import argparse

# 3rd-party
import torch
import matplotlib.pyplot as plt

# Project
from utils import process_image, imshow, cat_to_name, get_valid_device, load_model



###########
# Predict #
###########
def predict(image_path, model, device, topk=5):
    '''
        Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = torch.from_numpy(process_image(image_path)).float()
    image.requires_grad = False
    image = image.unsqueeze(0) # add batch dimension
    image = image.to(device)
    ps = torch.exp(model(image))
    top_p, top_categories = ps.topk(topk, dim=1)
    return top_p, top_categories


def solution(model, top_p, top_idx, category_names):
    """
        Makes solutions human readable
        
        - Converts probabilities to numpy array
        - Replaces the idx with human readable class names
    """
    top_p = top_p[0].cpu().detach().numpy()

    top_cats_names = []
    for idx in top_idx[0].cpu().detach().numpy():
        cat = model.idx_to_class[str(idx)]
        name = cat_to_name(cat, category_names)
        top_cats_names.append(name)

    return top_p, top_cats_names



########
# Main #
########
def main(image_path, checkpoint_path, category_names, top_k, gpu, display):
    """
        Display an image, predicts its most likely classifications
    """    
    device = get_valid_device(gpu)
    
    # verify image exists
    if not os.path.isfile(image_path):
        print(f'Image does not exists at path {image_path}')
        return -1
    if not os.path.isfile(category_names):
        print(f'Category names file does not exists at path {category_names}')
        return -1
    
    print('Loading model...', end='\r')  
    model = load_model(checkpoint_path)
    model.to(device)
    print('\033[K', end='\r')
    print('Loaded.', end='\r')

    top_p, top_idx = predict(image_path, model, device, top_k)

    probabilities, names = solution(model, top_p, top_idx, category_names)

    print('\033[K', end='\r')
    for p, name in zip(probabilities, names):
        print(f'{name.title():20s} with {p*100:8.1f} % probability')

    # Show image on display
    if display:
        image = process_image(image_path)
        imshow(image, title=top_cats_names[0])

        def bar_graph(x, y):
            plt.rcdefaults()
            fig, ax = plt.subplots()

            y_pos = np.arange(len(y))

            ax.barh(y_pos, x, align='center', color='blue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Probability')

            plt.show()

        # Display bar graph
        display(top_p, top_cats_names)
        bar_graph(top_p, top_cats_names)


if __name__ == '__main__':
    """
        Basic usage: python predict.py /path/to/image checkpoint
        Options:    
            Return top KK most likely classes: python predict.py input checkpoint --top_k 3
            Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
            Use GPU for inference: python predict.py input checkpoint --gpu
    """
    parser = argparse.ArgumentParser(
        description=
        """
        Image classifier prediction module
        
        Basic usage: python predict.py /path/to/image checkpoint
        Options:    
            Return top KK most likely classes: python predict.py input checkpoint --top_k 3
            Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
            Use GPU for inference: python predict.py input checkpoint --gpu
        """
    )

    parser.add_argument(action='store',
                        type=str,
                        dest='image_path',
                        help='Data directory')

    parser.add_argument(action='store',
                        type=str,
                        dest='checkpoint_path',
                        help='Checkpoint path')
    
    parser.add_argument('--category_names', action='store',
                        type=str,
                        default=None,
                        dest='category_names',
                        help='Load category names')
    
    parser.add_argument('--top_k', action='store',
                        type=int,
                        default=5,
                        dest='top_k',
                        help='List top K most likely classes')
    
    parser.add_argument('--gpu', action='store_true',
                        default=False,
                        dest='gpu',
                        help='Use GPU for inference')
    
    parser.add_argument('--display', action='store_true',
                        default=False,
                        dest='display',
                        help='Show image on graphics display')
    
    options = parser.parse_args()
    
    main(options.image_path, options.checkpoint_path, options.category_names, options.top_k, options.gpu, options.display)
