import time
import itertools
import random
import sys

import numpy as np
from PIL import Image
from skimage import img_as_float
from skimage.metrics import mean_squared_error as compare_mse
from tqdm import tqdm

import multiprocessing
from functools import partial

def shuffle_first_items(lst, i):
    if not i:
        return lst
    first_few = lst[:i]
    remaining = lst[i:]
    random.shuffle(first_few) 
    return first_few + remaining

def bound(low, high, value):
    return max(low, min(high, value))

class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0

    def update(self):
        self.counter += 1
        sys.stdout.write("Progress: %s%% %s" % (100 * self.counter / self.total, "\r"))
        sys.stdout.flush()

def img_mse(im1, im2):
    """Calculates the root mean square error (RSME) between two images"""
    try:
        return compare_mse(img_as_float(im1), img_as_float(im2))
    except ValueError:
        print(f'RMS issue, Img1: {im1.size[0]} {im1.size[1]}, Img2: {im2.size[0]} {im2.size[1]}')
        raise KeyboardInterrupt

def resize_box_aspect_crop_to_extent(img, target_aspect, centerpoint=None):
    width = img.size[0]
    height = img.size[1]
    if not centerpoint:
        centerpoint = (int(width / 2), int(height / 2))

    requested_target_x = centerpoint[0]
    requested_target_y = centerpoint[1]
    aspect = width / float(height)
    if aspect > target_aspect:
        # Then crop the left and right edges:
        new_width = int(target_aspect * height)
        new_width_half = int(new_width/2)
        target_x = bound(new_width_half, width-new_width_half, requested_target_x)
        left = target_x - new_width_half
        right = target_x + new_width_half
        resize = (left, 0, right, height)
    else:
        # ... crop the top and bottom: 
        new_height = int(width / target_aspect)
        new_height_half = int(new_height/2)
        target_y = bound(new_height_half, height-new_height_half, requested_target_y)
        top = target_y - new_height_half
        bottom = target_y + new_height_half
        resize = (0, top, width, bottom)
    return resize

def aspect_crop_to_extent(img, target_aspect, centerpoint=None):
    '''
    Crop an image to the desired perspective at the maximum size available.
    Centerpoint can be provided to focus the crop to one side or another - 
    eg just cut the left side off if interested in the right side.

    target_aspect = width / float(height)
    centerpoint = (width, height)
    '''
    resize = resize_box_aspect_crop_to_extent(img, target_aspect, centerpoint)
    return img.crop(resize)

class Config:
    def __init__(self, tile_ratio=1920/800, tile_width=50, enlargement=8, color_mode='RGB', blending=0):
        self.tile_ratio = tile_ratio # 2.4
        self.tile_width = tile_width # height/width of mosaic tiles in pixels
        self.enlargement = enlargement # mosaic image will be this many times wider and taller than original
        self.color_mode = color_mode # mosaic image will be this many times wider and taller than original
        self.blending = blending

    @property
    def tile_height(self):
        return int(self.tile_width / self.tile_ratio)

    @property
    def tile_size(self):
        return self.tile_width, self.tile_height # PIL expects (width, height)

class TileBox:
    """
    Container to import, process, hold, and compare all of the tiles 
    we have to make the mosaic with.
    """
    def __init__(self, tile_paths, config):
        self.config = config
        self.tiles = list()
        self.used_tiles = set()
        self.prepare_tiles_from_paths(tile_paths)
        
    def __process_tile(self, tile_path):
        with Image.open(tile_path) as i:
            img = i.copy()
        img = aspect_crop_to_extent(img, self.config.tile_ratio)
        large_tile_img = img.resize(self.config.tile_size, Image.ANTIALIAS).convert(self.config.color_mode)
        self.tiles.append(large_tile_img)
        return True

    def prepare_tiles_from_paths(self, tile_paths):
        print('Reading tiles from provided list...')
        progress = ProgressCounter(len(tile_paths))
        for tile_path in tile_paths:
            progress.update()
            if os.path.basename(tile_path)[0] != '.':
                self.__process_tile(tile_path)
        print('Processed tiles.')
        return True

    def best_tile_block_match(self, tile_block_original):
        match_results = [img_mse(t, tile_block_original) for i, t in enumerate(self.tiles) if i not in self.used_tiles]
        best_fit_tiles = np.argsort(match_results)[:5]
        best_fit_tile_index = np.random.choice(best_fit_tiles)
        return best_fit_tile_index

    def best_tile_from_block(self, tile_block_original, reuse=False):
        if len(self.tiles) == len(self.used_tiles):
            print('Ran out of images.')
            raise KeyboardInterrupt
        
        #start_time = time.time()
        i = self.best_tile_block_match(tile_block_original)
        #print("BLOCK MATCH took --- %s seconds ---" % (time.time() - start_time))
#        match = self.tiles[i].copy()
        if not reuse:
            self.used_tiles.add(i)
        match = i
        return match

class SourceImage:
    """Processing original image - scaling and cropping as needed."""
    def __init__(self, image_path, config):
        print('Processing main image...')
        self.image_path = image_path
        self.config = config

        with Image.open(self.image_path) as i:
            img = i.copy()
        w = img.size[0] * self.config.enlargement
        h = img.size[1]	* self.config.enlargement
        large_img = img.resize((w, h), Image.ANTIALIAS)
        w_diff = (w % self.config.tile_width)/2
        h_diff = (h % self.config.tile_height)/2
        
        # if necesary, crop the image slightly so we use a 
        # whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        self.image =  large_img.convert(self.config.color_mode)
        print('Main image processed.')

class MosaicImage:
    """Holder for the mosaic"""
    def __init__(self, original_img, target, config):
        self.config = config
        self.target = target
        # Lets just start with original image, scaled up, instead of a blank one
        self.image = original_img
        # self.image = Image.new(original_img.mode, original_img.size)
        self.x_tile_count = int(original_img.size[0] / self.config.tile_width)
        self.y_tile_count = int(original_img.size[1] / self.config.tile_height)
        self.total_tiles  = self.x_tile_count * self.y_tile_count
        print(f'Mosaic will be {self.x_tile_count:,} tiles wide and {self.y_tile_count:,} tiles high ({self.total_tiles:,} total).')
        self.np_image = np.array(self.image)

    def add_tile(self, tile, coords):
        """Adds the provided image onto the mosiac at the provided coords."""
        try:
            np_tile = np.array(tile)
            self.np_image[coords[1]:coords[3], coords[0]:coords[2], :] = self.config.blending*self.np_image[coords[1]:coords[3], coords[0]:coords[2], :] + (1.-self.config.blending)*np_tile
#            self.image.paste(tile, coords)
        except TypeError as e:
            print('Maybe the tiles are not the right size. ' + str(e))

    def save(self):
        Image.fromarray(np.uint8(self.np_image)).save(self.target)

def coords_from_middle(x_count, y_count, y_bias=1, shuffle_first=0, ):
    '''
    Lets start in the middle where we have more images.
    And we dont get "lines" where the same-best images
    get used at the start.

    y_bias - if we are using non-square coords, we can
        influence the order to be closer to the real middle.
        If width is 2x height, y_bias should be 2.

    shuffle_first - We can suffle the first X coords
        so that we dont use all the same-best images
        in the same spot -  in the middle

    from movies.mosaic_mem import coords_from_middle
    x = 10
    y = 10
    coords_from_middle(x, y, y_bias=2, shuffle_first=0)
    '''
    x_mid = int(x_count/2)
    y_mid = int(y_count/2)
    coords = list(itertools.product(range(x_count), range(y_count)))
    coords.sort(key=lambda c: abs(c[0]-x_mid)*y_bias + abs(c[1]-y_mid))
    coords = shuffle_first_items(coords, shuffle_first)
    return coords


def match_tile(coord, source_image, tile_box, config, reuse):
    x, y = coord
#    print(x,y)
    # Make a box for this sector
    box_crop = (x * config.tile_width, y * config.tile_height, (x + 1) * config.tile_width, (y + 1) * config.tile_height)

    # Get Original Image Data for this Sector
    comparison_block = source_image.crop(box_crop)

    # Get Best Image name that matches the Orig Sector image
    tile_match_i = tile_box.best_tile_from_block(comparison_block, reuse=reuse)
    
    return box_crop, tile_match_i


def create_mosaic(source_path, target, tile_ratio=1920/800, tile_width=75, enlargement=8, reuse=True, color_mode='RGB', tile_paths=None, shuffle_first=30, blending=0, njobs=1):
    """Forms an mosiac from an original image using the best
    tiles provided. This reads, processes, and keeps in memory
    a copy of the source image, and all the tiles while processing.

    Arguments:
    source_path -- filepath to the source image for the mosiac
    target -- filepath to save the mosiac
    tile_ratio -- height/width of mosaic tiles in pixels
    tile_width -- width of mosaic tiles in pixels
    enlargement -- mosaic image will be this many times wider and taller than the original
    reuse -- Should we reuse tiles in the mosaic, or just use each tile once?
    color_mode -- L for greyscale or RGB for color
    tile_paths -- List of filepaths to your tiles
    shuffle_first -- Mosiac will be filled out starting in the center for best effect. Also, 
        we will shuffle the order of assessment so that all of our best images aren't 
        necessarily in one spot.
    """
    config = Config(
        tile_ratio = tile_ratio,		# height/width of mosaic tiles in pixels
        tile_width = tile_width,		# height/width of mosaic tiles in pixels
        enlargement = enlargement,	    # the mosaic image will be this many times wider and taller than the original
        color_mode = color_mode,	    # L for greyscale or RGB for color
        blending = blending,
    )
    # Pull in and Process Original Image
    print('Setting Up Target image')
    source_image = SourceImage(source_path, config)

    # Setup Mosaic
    mosaic = MosaicImage(source_image.image, target, config)

    # Assest Tiles, and save if needed, returns directories where the small and large pictures are stored
    print('Assessing Tiles')
    tile_box = TileBox(tile_paths, config)
    
    progress = tqdm(total=mosaic.total_tiles)

    try:
        pool = multiprocessing.Pool(njobs)
        coords = coords_from_middle(mosaic.x_tile_count, mosaic.y_tile_count, y_bias=config.tile_ratio, shuffle_first=shuffle_first)
        print('Processing')

        results = []
        for box_crop, tile_match in pool.imap_unordered(partial(match_tile, source_image=source_image.image, tile_box=tile_box, config=config, reuse=reuse), coords, chunksize=max(1,mosaic.total_tiles//(100))):
            results.append((box_crop, tile_match))
            progress.update()
        progress.close()
    

        print('Pasting tiles to mosaic ...')
        for box_crop, tile_match_i in results:
            tile_match = tile_box.tiles[tile_match_i]
            mosaic.add_tile(tile_match, box_crop)
        mosaic.save()
    except KeyboardInterrupt:
        print('\nStopping, saving partial image...')
        for box_crop, tile_match_i in results:
            tile_match = tile_box.tiles[tile_match_i]
            mosaic.add_tile(tile_match, box_crop)
    finally:
        mosaic.save()


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('tiledirectory', type=str)
    parser.add_argument('--tile_ratio', type=float, default=1)
    parser.add_argument('--tile_width', type=int, default=50)
    parser.add_argument('--enlargement', type=int, default=2)
    parser.add_argument('--reuse', action='store_true', help='Enable the reuse of same tiles')
    parser.add_argument('--color_mode', type=str, default='RGB', help='RGB or L (for grayscale)')
    parser.add_argument('--blending', type=float, default=0, help='Value between 0 and 1. 0 no blending.')
    parser.add_argument('--njobs', type=int, default=1, help='Use more than one job to parallelize processing.')
    args = parser.parse_args()
    
    create_mosaic(args.source, args.target, tile_paths=[os.path.join(args.tiledirectory, f) for f in os.listdir(args.tiledirectory)], tile_ratio=args.tile_ratio, tile_width=args.tile_width, enlargement=args.enlargement, reuse=args.reuse, color_mode=args.color_mode, blending=args.blending, njobs=args.njobs)
