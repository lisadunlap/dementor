import os
import json
import argparse
import concurrent.futures
import threading
from tqdm import tqdm
from pathlib import Path

import serve.utils_vlm as utils_vlm
from serve.utils_general import get_from_cache, save_to_cache

def load_coco_annotations(annotations_file):
    """Load COCO annotations file."""
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def caption_image(img_id, img_path, model):
    """Caption a single image and return the result."""
    try:
        caption = utils_vlm.captioning(img_path, model=model)
        return img_id, caption, None
    except Exception as e:
        return img_id, None, str(e)

def process_split(image_dir, annotations_file, output_file, model="idefics", limit=None, skip_existing=True, num_workers=4):
    """Process a COCO dataset split and generate captions using multiple threads."""
    # Load existing results if available and skip_existing is True
    existing_captions = {}
    if skip_existing and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_captions = json.load(f)
        print(f"Loaded {len(existing_captions)} existing captions from {output_file}")
    
    # Load COCO annotations
    try:
        annotations = load_coco_annotations(annotations_file)
        images = annotations['images']
        print(f"Loaded {len(images)} images from annotations")
    except Exception as e:
        print(f"Error loading annotations: {e}")
        # If no annotations, get all image files directly
        images = [{'id': Path(img).stem, 'file_name': img} 
                 for img in os.listdir(image_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(images)} images in directory")
    
    # Limit the number of images if specified
    if limit:
        images = images[:limit]
    
    results = existing_captions.copy()
    results_lock = threading.Lock()
    save_lock = threading.Lock()
    
    # Function to save results to file
    def save_results():
        with save_lock:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Create a list of images to process (exclude already processed ones)
    to_process = []
    for img in images:
        img_id = str(img['id'])
        if skip_existing and img_id in existing_captions:
            continue
        img_path = os.path.join(image_dir, img['file_name'])
        to_process.append((img_id, img_path))
    
    print(f"Processing {len(to_process)} images with {num_workers} workers")
    
    # Process images with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_img = {
            executor.submit(caption_image, img_id, img_path, model): img_id 
            for img_id, img_path in to_process
        }
        
        # Process completed tasks with a progress bar
        completed = 0
        with tqdm(total=len(to_process)) as pbar:
            for future in concurrent.futures.as_completed(future_to_img):
                img_id, caption, error = future.result()
                
                with results_lock:
                    if error:
                        print(f"Error processing image {img_id}: {error}")
                    else:
                        results[img_id] = caption
                
                completed += 1
                pbar.update(1)
                
                # Save results periodically
                if completed % 100 == 0:
                    save_results()
    
    # Save final results
    save_results()
    print(f"Saved {len(results)} captions to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate captions for COCO dataset using Idefics")
    parser.add_argument("--coco_dir", type=str, default="/datasets/coco2014/current", 
                        help="Path to COCO dataset directory")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save generated captions")
    parser.add_argument("--model", type=str, default="idefics", 
                        choices=["idefics", "gpt-4o"], 
                        help="VLM model to use for captioning")
    parser.add_argument("--split", type=str, default="val2014", 
                        choices=["train2014", "val2014", "test2014"],
                        help="Dataset split to process")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit the number of images to process")
    parser.add_argument("--skip_existing", action="store_true", 
                        help="Skip already processed images")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set paths
    image_dir = os.path.join(args.coco_dir, args.split)
    annotations_file = None
    if args.split != "test2014":
        annotations_file = os.path.join(args.coco_dir, "annotations", f"captions_{args.split}.json")
    
    output_file = os.path.join(args.output_dir, f"{args.model}_{args.split}_captions.json")
    
    # Process the dataset split
    process_split(image_dir, annotations_file, output_file, 
                 model=args.model, limit=args.limit, 
                 skip_existing=args.skip_existing, num_workers=args.workers)

if __name__ == "__main__":
    main()
