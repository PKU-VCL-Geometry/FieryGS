import os
import argparse
from utils.vlm_utils import get_image_files, Qwen, GPT4V, GPTo3, GPTtools, GPTtools_supp


def query_vlm(base_path, case_name, vlm_type = "qwen"):
    # input_image_path = os.path.join(base_path, case_name)
    input_image_path = os.path.join(base_path, case_name)
    image_files = get_image_files(input_image_path)

    # material_list = "wood, metal, plastic, glass, fabric, foam, food, ceramic, paper, leather"
    # new material
    material_list = "wood, sand, metal, plastic, glass, fabric, foam, food, ceramic, paper, leather, plant, stone, cement, concrete, soil, clay, composite, sky"
    material_list = material_list.split(", ")
    material_library = "{" + ", ".join(material_list) + "}"

    # prompt = f"""Provided a picture. The left image is the original picture of the object (Original Image), and the middle image is a partial segmentation diagram (Mask Overlay), mask is in red. The right image is a partial of the object. 
    # Based on the image, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the object is and the material of the object to predict the hardness of the part. Choose whether to use Shore A hardness or Shore D hardness depending on the material. You may provide a range of values for hardness instead of a single value. 

    # Format Requirement:
    # You must provide your answer as a (brief caption of the part, material of the part, hardness, Shore A/D) pair. Do not include any other text in your answer, as it will be parsed by a code script later. 
    # common material library: {material_library}. 
    # Your answer must look like: caption, material, hardness low-high, <Shore A or Shore D>. 
    # The material type must be chosen from the above common material library. Make sure to use Shore A or Shore D hardness, not Mohs hardness."""
    
    # fire gaussian
    # prompt = f"""Provided a picture. The left image is the original picture of the object (Original Image), and the middle image is a partial segmentation diagram (Mask Overlay), mask is in blue. The right image is a partial of the object. 
    # Based on the image, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the object is and the material of the object to determine whether the material is burnable.

    # Format Requirement:
    # You must provide your answer as a (brief caption of the part, material of the part, burnable/unburnable) pair. Do not include any other text in your answer, as it will be parsed by a code script later. 
    # common material library: {material_library}. 
    # Your answer must look like: caption, material, burnable/unburnable. 
    # The material type must be chosen from the above common material library."""
    
    # update
    # prompt_mul = f"""Provided a picture composed of four images arranged from left to right: 1. Original Image: The original photo of the entire object. 2. Parent Part: A segmentation overlay showing the larger part that contains the current component. 3. Mask Overlay: A segmentation overlay showing the current component in blue with a red bounding box. 4. Part Image: A cropped and centered view of the segmented component. The fourth image (Part Image) provides the clearest view of the target part and is the most important image for identifying it. 
    # Based on the image, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the object is and the material of the object to determine whether the material is burnable.

    # Format Requirement:
    # You must provide your answer as a (brief caption of the part, material of the part, burnable/unburnable) pair. Do not include any other text in your answer, as it will be parsed by a code script later. 
    # common material library: {material_library}. 
    # Your answer must look like: caption, material, burnable/unburnable. 
    # The material type must be chosen from the above common material library."""
    
    # prompt_sin = f"""Provided a picture. The left image is the original picture of the object (Original Image), and the middle image is a partial segmentation diagram (Mask Overlay) -- the target mask is shown in blue, and the corresponding region is outlined with a red bounding box. The right image is a cropped/zoomed view of that object part. The right image (Part Image) provides the clearest view of the target part and is the most important image for identifying it.
    # Based on the picture, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the object is and the material of the object to determine whether the material is burnable.

    # Format Requirement:
    # You must provide your answer as a (brief caption of the part, material of the part, burnable/unburnable) pair. Do not include any other text in your answer, as it will be parsed by a code script later. 
    # common material library: {material_library}. 
    # Your answer must look like: caption, material, burnable/unburnable. 
    # The material type must be chosen from the above common material library."""
    
    prompt_sin = f"""Provided a picture composed of three images arranged from left to right: 1. **Original Image**: The original photo of the entire scene. 2. **Mask Overlay**: A segmentation overlay highlighting the part of interest in blue, with a red bounding box. (This may be either the object or the background.) 3. **Part Image**: A cropped and centered view showing only the segmented part.
    
    You may use all three images to understand what the part is and identify whether the segmented part is an object or the background. The third image (Part Image) provides the clearest visual of the target part, but context from the first and second images may also be useful for identification.
    
    Based on the picture, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the scene is and the material of the part to determine whether the part is burnable.

    Format Requirement:
    You must provide your answer as a (brief caption of the part, material of the part, burnable/unburnable) pair. Do not include any other text in your answer, as it will be parsed by a code script later.  
    common material library: {material_library}.  Under most circumstances, things made of plastic and plants are burnable.
    Your answer must look like: caption, material, burnable/unburnable.  
    The material type must be chosen from the above common material library."""
    
    prompt_sin_supp = f"""Provided a picture composed of three images arranged from left to right:
1. **Original Image**: The original photo of the entire scene.
2. **Mask Overlay**: A segmentation overlay highlighting the part of interest in blue, with a red bounding box. (This may be either the object or the background.)
3. **Part Image**: A cropped and centered view showing only the segmented part.

You may use all three images to understand what the part is and identify whether the segmented part is an object or the background.
The third image (Part Image) provides the clearest visual of the target part, but context from the first and second images may also be useful for identification.

Based on the picture, firstly provide a brief caption of the part.
Secondly, describe what the part is made of (provide the major one).
Thirdly, we combine what the scene is and the material of the part to determine whether the part is burnable.

Finally, you must provide:
- the thermal diffusivity ratio (i.e., the material's thermal diffusivity divided by that of wood);
- the typical smoke color when this material is burned.

Format Requirement:
You must provide your answer as a 5-part tuple:
(caption of the part, material of the part, burnable/unburnable, thermal diffusivity ratio vs. wood, typical smoke color)

Do not include any other text in your answer, as it will be parsed by a code script later.

common material library: {material_library}.
Under most circumstances, things made of plastic and plants are burnable, and sky is unburnable.
Use typical values of thermal diffusivity for each material.
Use approximate known smoke colors (e.g., black, white, gray, yellowish, etc.) based on typical combustion results.
"""

#     prompt_sin = f"""Provided a picture composed of three images arranged from left to right:  
# 1. **Original Image**: The original photo of the entire scene.  
# 2. **Mask Overlay**: A segmentation overlay highlighting the part of interest in blue, with a red bounding box. (This may include either an object or the background.)  
# 3. **Part Image**: A cropped and centered view showing only the segmented part.  

# You may use all three images to understand what the part is and determine whether the segmented part represents an object or the background area. The third image (Part Image) provides the clearest view of the target part, but the first two images help provide scene context.

# Based on the picture, first provide a **brief caption** of the part.  
# Then, describe **what the part is made of** (choose the dominant material).  
# Finally, **determine whether the material is burnable or unburnable**, considering the type of material.

# **Output Format:**  
# - First, clearly explain the reason why the material is considered burnable or unburnable.  
# - Then, on a new line, provide your answer in the format:  
#   `caption, material, burnable/unburnable`  

# **Common material library:**  
# {material_library}  

# Example:  
# Because wood is an organic material that catches fire easily under normal indoor conditions, especially when untreated.  
# `wooden block, wood, burnable`

# WARNING: Do not include any other text. Only output the explanation and the result line, exactly as required.
# """

    
    prompt_mul = f"""Provided a picture composed of four images arranged from left to right: 1. **Original Image**: The original photo of the scene, with a yellow highlight indicating the larger part (parent) to which the target component belongs. 2. **Parent Part**: A segmentation overlay showing the larger part that contains the target component. 3. **Mask Overlay**: A segmentation overlay showing the target component in blue, with a red bounding box. 4. **Part Image**: A cropped and centered view showing only the segmented component.

    All four images provide helpful context. The fourth image (Part Image) gives the clearest visual of the component. Please consider whether the current component is consistent with the rest of the parent part in terms of material.

    Based on the picture, firstly provide a brief caption of the target component. Secondly, describe what the component is made of (provide the major one). Finally, we combine what the component is and the material of the component to determine whether the component is burnable.

    Format Requirement:
    You must provide your answer as a (brief caption of the component, material of the component, burnable/unburnable) pair. Do not include any other text in your answer, as it will be parsed by a code script later.  
    common material library: {material_library}. Under most circumstances, things made of plastic and plants are burnable and sky is unburnable.
    Your answer must look like: caption, material, burnable/unburnable.  
    The material type must be chosen from the above common material library."""
    
    prompt_mul_supp = f"""Provided a picture composed of four images arranged from left to right:
1. **Original Image**: The original photo of the scene, with a yellow highlight indicating the larger part (parent) to which the target component belongs.
2. **Parent Part**: A segmentation overlay showing the larger part that contains the target component.
3. **Mask Overlay**: A segmentation overlay showing the target component in blue, with a red bounding box.
4. **Part Image**: A cropped and centered view showing only the segmented component.

All four images provide helpful context. The fourth image (Part Image) gives the clearest visual of the component.
Please consider whether the current component is consistent with the rest of the parent part in terms of material.

Based on the picture, firstly provide a brief caption of the target component.
Secondly, describe what the component is made of (provide the major one).
Thirdly, we combine what the component is and the material of the component to determine whether the component is burnable.

Finally, you must also provide:
- the thermal diffusivity ratio (i.e., the component’s material thermal diffusivity divided by that of wood);
- the typical smoke color when this material burns.

Format Requirement:
You must provide your answer as a 5-part tuple:
(caption of the component, material of the component, burnable/unburnable, thermal diffusivity ratio vs. wood, typical smoke color)

Do not include any other text in your answer, as it will be parsed by a code script later.

common material library: {material_library}.
Under most circumstances, things made of plastic and plants are burnable, and sky is unburnable.
Use typical values of thermal diffusivity for each material.
Use approximate known smoke colors (e.g., black, white, gray, yellowish, etc.) based on typical combustion results.
"""

    
    #update
    # prompt = f"""You are given three images aligned left→right:
    #             1. Original Image – full view of the object.
    #             2. Mask Overlay – a partial segmentation diagram with a mask in blue and bounding box in red.
    #             3. A partial of the object – a zoom‑in of the masked region.

    #             Your task:
    #             1. Write a brief caption of the highlighted part.
    #             2. Identify the dominant material of that part (choose exactly one from the ,material library below).
    #             3. Decide whether the material is burnable or unburnable.

    #             Output format – one single line, comma‑separated (no spaces!):
    #             <caption>,<material>,<burnable|unburnable>
    #             Do not output anything else. If uncertain, make your best guess.

    #             Material library (choose one; case‑sensitive): {material_library}

    #             Examples:
    #             handle,metal,unburnable
                
    #             Now look at the images and respond."""

    output_file = f'{case_name}_{vlm_type}.txt'
    results_file_path = os.path.join(base_path, case_name, output_file)
    case_msg = ""
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    if case_name == 'gpt_input_single':
        prompt = prompt_sin
        # prompt = None
    elif case_name == 'gpt_input_mul':
        prompt = prompt_mul
    else:
        prompt = prompt_sin_supp
        
    if prompt != None:
        with open(results_file_path, 'w') as file:
            for image_file in image_files:
                if vlm_type == 'qwen':
                    message = str(Qwen(image_file, prompt))
                elif vlm_type == 'gpt4v':
                    message = str(GPT4V(image_file, prompt))
                elif vlm_type == 'gpto3':
                    message = str(GPTo3(image_file, prompt))
                elif vlm_type == "gpttool":
                    message = str(GPTtools(image_file, prompt))
                elif vlm_type == "supp":
                    message = str(GPTtools_supp(image_file, prompt))
                        
                # except KeyError as e:
                #     message = "error,-1"
                # except Exception as e:
                #     message = "error,-1"
                
                # message = message['choices'][0]['message']['content']  
                write_msg = image_file + "," + message
                print(write_msg)
                case_msg += case_msg
                file.write(f"{write_msg}\n")
                file.flush()

    print("Messages have been written to", results_file_path)


def run_vlm(base_path, vlm_type):
    all_cases = os.listdir(base_path)
    for case_name in all_cases:
        # if case_name == "gpt_input_0.8_bbox_":
        query_vlm(base_path, case_name, vlm_type=vlm_type)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('--vlm', type=str, default="qwen", help="gpt4v, qwen, gpto3")
    parser.add_argument('--dataset_path', type=str, default="gp_cases_dirs")
    args = parser.parse_args()
    run_vlm(args.dataset_path, args.vlm)

