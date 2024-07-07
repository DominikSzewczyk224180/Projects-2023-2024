# Written by Kian & ChatGPT
# Import standard libraries
import argparse
import glob
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, call, patch

# Import third-party libraries
import cv2
import inquirer
import numpy as np
from azureml.core import Dataset
from azureml.data.datapath import DataPath

# Add the parent directory to the sys.path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom functions
from process_data import(
    set_working_directory,
    select_directory,
    check_directory_and_handle_contents,
    get_image_paths,
    get_masks,
    get_image_paths_and_masks,
    select_masks,
    validate_image_paths,
    split_data,
    create_directories,
    crop_dimensions,
    padder,
    save_patches,
    process_images,
    upload_directory,
    main
)

class TestYourScriptName(unittest.TestCase):
    def setUp(self):
        # Set up a dummy grayscale image with a larger circle
        self.img = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(self.img, (250, 250), 220, (255), -1)
        self.patch_size = 256

    @patch('os.path.abspath')
    @patch('os.path.dirname')
    @patch('os.chdir')
    @patch('os.getcwd')
    @patch('process_data.logger')
    def test_set_working_directory(self, mock_logger, mock_getcwd, mock_chdir, mock_dirname, mock_abspath):
        # Mock values
        mock_abspath.return_value = '/path/to/current/script'
        mock_dirname.side_effect = lambda path: os.path.split(path)[0]
        
        # Expected root directory three levels up
        expected_root_directory = '/path'
        
        # Call the function
        set_working_directory()
        
        # Assertions
        mock_abspath.assert_called_once_with(sys.argv[0])
        self.assertEqual(mock_dirname.call_count, 3)
        mock_chdir.assert_called_once_with(expected_root_directory)
        mock_getcwd.assert_called_once()
        mock_logger.info.assert_called_once_with("Working directory set to: %s", mock_getcwd.return_value)

    @patch('os.path.abspath')
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.join', side_effect=lambda *args: os.path.sep.join(args))
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_select_directory_default_mode(self, mock_prompt, mock_logger, mock_join, mock_isdir, mock_listdir, mock_abspath):
        mock_abspath.return_value = '/abs/path/data/raw'
        mock_listdir.return_value = ['dir1', 'dir2', 'dir3']
        mock_isdir.side_effect = lambda x: x in ['/abs/path/data/raw/dir1', '/abs/path/data/raw/dir2', '/abs/path/data/raw/dir3']
        
        result = select_directory(mode='default')
        
        mock_abspath.assert_called_once_with('data/raw')
        mock_listdir.assert_called_once_with('/abs/path/data/raw')
        self.assertEqual(mock_isdir.call_count, 3)
        self.assertEqual(result, '/abs/path/data/raw/dir1')
        mock_logger.info.assert_called_once_with("Default mode: Automatically selected /abs/path/data/raw/dir1")
        
    @patch('os.path.abspath')
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.join', side_effect=lambda *args: os.path.sep.join(args))
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_select_directory_custom_mode(self, mock_prompt, mock_logger, mock_join, mock_isdir, mock_listdir, mock_abspath):
        mock_abspath.return_value = '/abs/path/data/raw'
        mock_listdir.return_value = ['dir1', 'dir2', 'dir3']
        mock_isdir.side_effect = lambda x: x in ['/abs/path/data/raw/dir1', '/abs/path/data/raw/dir2', '/abs/path/data/raw/dir3']
        mock_prompt.return_value = {'directory': 'dir2'}
        
        result = select_directory(mode='custom')
        
        mock_abspath.assert_called_once_with('data/raw')
        mock_listdir.assert_called_once_with('/abs/path/data/raw')
        self.assertEqual(mock_isdir.call_count, 3)
        mock_prompt.assert_called_once()
        self.assertEqual(result, '/abs/path/data/raw/dir2')
        mock_logger.info.assert_called_once_with("You have selected: /abs/path/data/raw/dir2")
        
    @patch('os.path.abspath')
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.join', side_effect=lambda *args: os.path.sep.join(args))
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_select_directory_no_directories(self, mock_prompt, mock_logger, mock_join, mock_isdir, mock_listdir, mock_abspath):
        mock_abspath.return_value = '/abs/path/data/raw'
        mock_listdir.return_value = []
        mock_isdir.side_effect = lambda x: False
        
        result = select_directory(mode='custom')
        
        mock_abspath.assert_called_once_with('data/raw')
        mock_listdir.assert_called_once_with('/abs/path/data/raw')
        self.assertEqual(result, '/abs/path/data/raw')
        mock_logger.warning.assert_called_once_with("No directories found in /abs/path/data/raw")
        
    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_default_mode(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [['img_file'], []]
        result = check_directory_and_handle_contents('data/raw/some_dir', 'default', 'local')
        
        self.assertEqual(result, (True, 'd', 'data/processed/some_dir'))
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call('Deleting directory contents...')
        mock_logger.info.assert_any_call("All contents in the destination directory have been deleted.")
        mock_rmtree.assert_called_once_with('data/processed/some_dir')
        
    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_custom_mode_delete(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [['img_file'], []]
        mock_prompt.return_value = {'action': 'd'}
        result = check_directory_and_handle_contents('data/raw/some_dir', 'custom', 'local')
        
        self.assertEqual(result, (True, 'd', 'data/processed/some_dir'))
        mock_prompt.assert_called_once()
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call('Deleting directory contents...')
        mock_logger.info.assert_any_call("All contents in the destination directory have been deleted.")
        mock_rmtree.assert_called_once_with('data/processed/some_dir')
        
    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_custom_mode_add(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [['img_file'], []]
        mock_prompt.return_value = {'action': 'a'}
        result = check_directory_and_handle_contents('data/raw/some_dir', 'custom', 'local')
        
        self.assertEqual(result, (True, 'a', 'data/processed/some_dir'))
        mock_prompt.assert_called_once()
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call("You chose to add non-existing files only (not available in early-access).")
        mock_rmtree.assert_not_called()
        
    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_no_files(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [[], []]
        result = check_directory_and_handle_contents('data/raw/some_dir', 'custom', 'local')
        
        self.assertEqual(result, (True, 'a', 'data/processed/some_dir'))
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_any_call("Destination directory is empty, continuing with the process.")
        mock_rmtree.assert_not_called()

    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_default_mode(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [['img_file'], []]
        result = check_directory_and_handle_contents('data/raw/some_dir', 'default', 'local')
        
        self.assertEqual(result, (True, 'd', 'data/processed/some_dir'))
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call('Deleting directory contents...')
        mock_logger.info.assert_any_call("All contents in the destination directory have been deleted.")
        mock_rmtree.assert_called_once_with('data/processed/some_dir')
        
    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_custom_mode_delete(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [['img_file'], []]
        mock_prompt.return_value = {'action': 'd'}
        result = check_directory_and_handle_contents('data/raw/some_dir', 'custom', 'local')
        
        self.assertEqual(result, (True, 'd', 'data/processed/some_dir'))
        mock_prompt.assert_called_once()
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call('Deleting directory contents...')
        mock_logger.info.assert_any_call("All contents in the destination directory have been deleted.")
        mock_rmtree.assert_called_once_with('data/processed/some_dir')
        
    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_custom_mode_add(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [['img_file'], []]
        mock_prompt.return_value = {'action': 'a'}
        result = check_directory_and_handle_contents('data/raw/some_dir', 'custom', 'local')
        
        self.assertEqual(result, (True, 'a', 'data/processed/some_dir'))
        mock_prompt.assert_called_once()
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call("You chose to add non-existing files only (not available in early-access).")
        mock_rmtree.assert_not_called()
        
    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_no_files(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [[], []]
        result = check_directory_and_handle_contents('data/raw/some_dir', 'custom', 'local')
        
        self.assertEqual(result, (True, 'a', 'data/processed/some_dir'))
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_any_call("Destination directory is empty, continuing with the process.")
        mock_rmtree.assert_not_called()

    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_no_masks(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [['img_file'], []]
        mock_prompt.return_value = {'action': 'd'}
        result = check_directory_and_handle_contents('data/raw/some_dir', 'custom', 'cloud')
        
        self.assertEqual(result, (True, 'd', 'data/processed/temp/some_dir'))
        mock_prompt.assert_called_once()
        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_any_call('Deleting directory contents...')
        mock_logger.info.assert_any_call("All contents in the destination directory have been deleted.")
        mock_rmtree.assert_called_once_with('data/processed/temp/some_dir')
        
    @patch('os.path.isdir')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('shutil.rmtree')
    @patch('process_data.logger')
    @patch('process_data.inquirer.prompt')
    def test_check_directory_and_handle_contents_error_deleting_contents(self, mock_prompt, mock_logger, mock_rmtree, mock_listdir, mock_makedirs, mock_exists, mock_isdir):
        mock_listdir.side_effect = [['img_file'], []]
        mock_prompt.return_value = {'action': 'd'}
        mock_rmtree.side_effect = Exception("Mocked deletion error")
        
        result = check_directory_and_handle_contents('data/raw/some_dir', 'custom', 'local')
        
        self.assertEqual(result, (False, None, 'data/processed/some_dir'))
        mock_logger.error.assert_called_once_with("Error deleting contents: %s", mock_rmtree.side_effect)
        mock_rmtree.assert_called_once_with('data/processed/some_dir')


    @patch('glob.glob')
    @patch('os.path.join')
    @patch('process_data.logger')
    def test_get_image_paths(self, mock_logger, mock_path_join, mock_glob):
        img_dir = '/path/to/images'
        extensions = ['*.png', '*.jpg', '*.jpeg']

        # Mock the behavior of os.path.join
        mock_path_join.side_effect = lambda img_dir, ext: f'{img_dir}/{ext}'

        # Mock the behavior of glob.glob
        mock_glob.side_effect = [
            ['img1.png', 'img2.png'],  # Results for *.png
            ['img3.jpg'],             # Results for *.jpg
            []                        # Results for *.jpeg
        ]

        expected_paths = ['img1.png', 'img2.png', 'img3.jpg']
        result = get_image_paths(img_dir, extensions)

        self.assertEqual(result, expected_paths)

        # Ensure os.path.join was called correctly
        for ext in extensions:
            mock_path_join.assert_any_call(img_dir, ext)
        
        # Ensure glob.glob was called correctly
        for ext in extensions:
            mock_glob.assert_any_call(f'{img_dir}/{ext}')
        
        # Ensure the logger was called correctly
        mock_logger.info.assert_any_call("Found %d images with extension %s", 2, '*.png')
        mock_logger.info.assert_any_call("Found %d images with extension %s", 1, '*.jpg')
        mock_logger.info.assert_any_call("Total images found: %d", 3)

    @patch('glob.glob')
    @patch('os.path.join')
    @patch('process_data.logger')
    def test_get_image_paths_default_extensions(self, mock_logger, mock_path_join, mock_glob):
        img_dir = '/path/to/images'

        # Mock the behavior of os.path.join
        mock_path_join.side_effect = lambda img_dir, ext: f'{img_dir}/{ext}'

        # Mock the behavior of glob.glob
        mock_glob.side_effect = [
            ['img1.png'],
            ['img2.jpg'],
            ['img3.jpeg'],
            ['img4.bmp'],
            ['img5.tif'],
            ['img6.tiff']
        ]

        expected_paths = ['img1.png', 'img2.jpg', 'img3.jpeg', 'img4.bmp', 'img5.tif', 'img6.tiff']
        result = get_image_paths(img_dir)

        self.assertEqual(result, expected_paths)

        default_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        # Ensure os.path.join was called correctly
        for ext in default_extensions:
            mock_path_join.assert_any_call(img_dir, ext)
        
        # Ensure glob.glob was called correctly
        for ext in default_extensions:
            mock_glob.assert_any_call(f'{img_dir}/{ext}')

        # Ensure the logger was called correctly
        for ext, count in zip(default_extensions, [1, 1, 1, 1, 1, 1]):
            mock_logger.info.assert_any_call("Found %d images with extension %s", count, ext)
        mock_logger.info.assert_any_call("Total images found: %d", 6)


    @patch('os.path.basename')
    @patch('os.path.dirname')
    @patch('os.listdir')
    def test_get_masks_with_extensions(self, mock_listdir, mock_dirname, mock_basename):
        fpath = '/path/to/images/image1.png'
        mask_extensions = ['mask1', 'mask2']

        mock_basename.return_value = 'image1.png'
        mock_dirname.return_value = '/path/to/images'
        mock_listdir.return_value = ['image1_mask1', 'image1_mask2', 'image1_mask3']

        expected_masks = ['image1_mask1', 'image1_mask2']
        result = get_masks(fpath, mask_extensions)

        self.assertEqual(result, expected_masks)

        mock_basename.assert_called_once_with(fpath)
        mock_dirname.assert_called_once_with(fpath)
        mock_listdir.assert_called_once_with('/path/to/masks')

    @patch('os.path.basename')
    @patch('os.path.dirname')
    @patch('os.listdir')
    def test_get_masks_without_extensions(self, mock_listdir, mock_dirname, mock_basename):
        fpath = '/path/to/images/image1.png'

        mock_basename.return_value = 'image1.png'
        mock_dirname.return_value = '/path/to/images'
        mock_listdir.return_value = ['image1_mask1', 'image1_mask2', 'image1_mask3']

        expected_masks = ['image1_mask1', 'image1_mask2', 'image1_mask3']
        result = get_masks(fpath)

        self.assertEqual(result, expected_masks)

        mock_basename.assert_called_once_with(fpath)
        mock_dirname.assert_called_once_with(fpath)
        mock_listdir.assert_called_once_with('/path/to/masks')



    @patch('process_data.get_masks')
    @patch('process_data.get_image_paths')
    @patch('process_data.logger')
    def test_get_image_paths_and_masks_default_mode(self, mock_logger, mock_get_image_paths, mock_get_masks):
        img_dir = '/path/to/images'
        mode = 'default'
        mock_get_image_paths.return_value = ['img1.png', 'img2.png']
        mock_get_masks.return_value = ['img1_mask1', 'img1_mask2']

        expected_img_paths = ['img1.png', 'img2.png']
        expected_mask_extensions = ['_mask1', '_mask2']

        result_img_paths, result_mask_extensions = get_image_paths_and_masks(img_dir, mode)

        self.assertEqual(result_img_paths, expected_img_paths)
        self.assertEqual(result_mask_extensions, expected_mask_extensions)

        mock_get_image_paths.assert_called_once_with(img_dir)
        mock_get_masks.assert_called_once_with('img1.png')
        mock_logger.info.assert_any_call("The following masks have been found for img1: ['_mask1', '_mask2'].")

    @patch('sys.exit')
    @patch('process_data.inquirer.prompt')
    @patch('process_data.get_masks')
    @patch('process_data.get_image_paths')
    @patch('process_data.logger')
    def test_get_image_paths_and_masks_custom_mode_cancel(self, mock_logger, mock_get_image_paths, mock_get_masks, mock_inquirer_prompt, mock_sys_exit):
        img_dir = '/path/to/images'
        mode = 'custom'
        mock_get_image_paths.return_value = ['img1.png', 'img2.png']
        mock_get_masks.return_value = ['img1_mask1', 'img1_mask2']
        mock_inquirer_prompt.return_value = {'action': 'cancel'}

        get_image_paths_and_masks(img_dir, mode)

        mock_get_image_paths.assert_called_once_with(img_dir)
        mock_get_masks.assert_called_once_with('img1.png')
        mock_inquirer_prompt.assert_called_once()
        mock_sys_exit.assert_called_once()
        mock_logger.warning.assert_any_call("Please check the naming convention of the provided data to ensure it is what the system expects /ref to documentation.")

    @patch('process_data.get_masks')
    @patch('process_data.get_image_paths')
    @patch('process_data.logger')
    def test_get_image_paths_and_masks_no_images_found(self, mock_logger, mock_get_image_paths, mock_get_masks):
        img_dir = '/path/to/images'
        mode = 'default'
        mock_get_image_paths.return_value = []

        expected_img_paths = []
        expected_mask_extensions = []

        result_img_paths, result_mask_extensions = get_image_paths_and_masks(img_dir, mode)

        self.assertEqual(result_img_paths, expected_img_paths)
        self.assertEqual(result_mask_extensions, expected_mask_extensions)

        mock_get_image_paths.assert_called_once_with(img_dir)
        mock_get_masks.assert_not_called()
        mock_logger.warning.assert_any_call("No images found in the directory: %s", img_dir)



    @patch('process_data.inquirer.prompt')
    def test_select_masks_default_mode(self, mock_inquirer_prompt):
        mask_extensions = ['mask1', 'mask2', 'mask3']
        mode = 'default'

        expected_masks = mask_extensions
        result = select_masks(mask_extensions, mode)

        self.assertEqual(result, expected_masks)
        mock_inquirer_prompt.assert_not_called()

    @patch('process_data.inquirer.prompt')
    def test_select_masks_custom_mode(self, mock_inquirer_prompt):
        mask_extensions = ['mask1', 'mask2', 'mask3']
        mode = 'custom'

        mock_inquirer_prompt.return_value = {'masks': ['mask1', 'mask3']}

        expected_masks = ['mask1', 'mask3']
        result = select_masks(mask_extensions, mode)

        self.assertEqual(result, expected_masks)
        mock_inquirer_prompt.assert_called_once()

        # Check that inquirer.prompt was called with the correct question
        called_args = mock_inquirer_prompt.call_args[0][0]
        self.assertEqual(len(called_args), 1)
        self.assertIsInstance(called_args[0], inquirer.Checkbox)
        self.assertEqual(called_args[0].name, 'masks')
        self.assertEqual(called_args[0].message, 'Select the masks you want to process:')
        self.assertEqual(called_args[0].choices, mask_extensions)
        self.assertEqual(called_args[0].default, mask_extensions)

    @patch('process_data.sys.exit')
    @patch('process_data.inquirer.prompt')
    @patch('process_data.logger')
    @patch('process_data.os.listdir')
    def test_validate_image_paths_no_missing_masks(self, mock_listdir, mock_logger, mock_inquirer_prompt, mock_sys_exit):
        img_paths = ['/path/to/images/img1.png', '/path/to/images/img2.png']
        selected_masks = ['_mask1', '_mask2']
        img_dir = '/path/to/images'
        mode = 'default'

        masks_dir_files = ['img1_mask1', 'img1_mask2', 'img2_mask1', 'img2_mask2']
        mock_listdir.return_value = masks_dir_files

        expected_valid_paths = img_paths
        result = validate_image_paths(img_paths.copy(), selected_masks, img_dir, mode)

        self.assertEqual(result, expected_valid_paths)
        mock_listdir.assert_called_once_with('/path/to/masks')
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_any_call("No missing masks detected.")
        mock_inquirer_prompt.assert_not_called()
        mock_sys_exit.assert_not_called()

    @patch('process_data.sys.exit')
    @patch('process_data.inquirer.prompt')
    @patch('process_data.logger')
    @patch('process_data.os.listdir')
    def test_validate_image_paths_missing_masks_default_mode(self, mock_listdir, mock_logger, mock_inquirer_prompt, mock_sys_exit):
        img_paths = ['/path/to/images/img1.png', '/path/to/images/img2.png']
        selected_masks = ['_mask1', '_mask2']
        img_dir = '/path/to/images'
        mode = 'default'

        masks_dir_files = ['img1_mask1','img1_mask2', 'img2_mask1']
        mock_listdir.return_value = masks_dir_files

        expected_valid_paths = ['/path/to/images/img1.png']
        result = validate_image_paths(img_paths.copy(), selected_masks, img_dir, mode)

        self.assertEqual(result, expected_valid_paths)
        mock_listdir.assert_called_once_with('/path/to/masks')
        mock_logger.warning.assert_any_call("Missing masks for 'img2': ['_mask2'].")
        mock_inquirer_prompt.assert_not_called()
        mock_sys_exit.assert_not_called()


    @patch('process_data.sys.exit')
    @patch('process_data.inquirer.prompt')
    @patch('process_data.logger')
    @patch('process_data.os.listdir')
    def test_validate_image_paths_missing_masks_interactive_mode_continue(self, mock_listdir, mock_logger, mock_inquirer_prompt, mock_sys_exit):
        img_paths = ['/path/to/images/img1.png', '/path/to/images/img2.png']
        selected_masks = ['_mask1', '_mask2']
        img_dir = '/path/to/images'
        mode = 'custom'

        masks_dir_files = ['img1_mask1', 'img2_mask1', 'img1_mask2']
        mock_listdir.return_value = masks_dir_files
        mock_inquirer_prompt.return_value = {'action': 'continue'}

        expected_valid_paths = ['/path/to/images/img1.png']
        result = validate_image_paths(img_paths.copy(), selected_masks, img_dir, mode)

        self.assertEqual(result, expected_valid_paths)
        mock_listdir.assert_called_once_with('/path/to/masks')
        mock_logger.warning.assert_any_call("Missing masks for 'img2': ['_mask2'].")
        mock_inquirer_prompt.assert_called_once()
        mock_sys_exit.assert_not_called()

    @patch('process_data.sys.exit')
    @patch('process_data.inquirer.prompt')
    @patch('process_data.logger')
    @patch('process_data.os.listdir')
    def test_validate_image_paths_missing_masks_custom_mode_cancel(self, mock_listdir, mock_logger, mock_inquirer_prompt, mock_sys_exit):
        img_paths = ['/path/to/images/img1.png', '/path/to/images/img2.png']
        selected_masks = ['_mask1', '_mask2']
        img_dir = '/path/to/images'
        mode = 'custom'

        masks_dir_files = ['img1_mask1', 'img2_mask1']
        mock_listdir.return_value = masks_dir_files
        mock_inquirer_prompt.return_value = {'action': 'cancel'}


        validate_image_paths(img_paths.copy(), selected_masks, img_dir, mode)

        mock_listdir.assert_called_once_with('/path/to/masks')
        mock_logger.warning.assert_any_call("Missing masks for 'img2': ['_mask2'].")
        mock_inquirer_prompt.assert_called_once()
        mock_sys_exit.assert_called_once()
        mock_logger.warning.assert_any_call("Please check the naming convention of the provided data to ensure it is what the system expects /ref to documentation.")






    @patch('process_data.train_test_split')
    @patch('process_data.logger')
    def test_split_data(self, mock_logger, mock_train_test_split):
        img_paths = ['/path/to/images/img1.png', '/path/to/images/img2.png', '/path/to/images/img3.png', '/path/to/images/img4.png', '/path/to/images/img5.png']
        train_size = 0.6
        val_size = 0.2
        random_state = 42

        mock_train_test_split.side_effect = [
            (['img1.png', 'img2.png', 'img3.png'], ['img4.png', 'img5.png']),  # First split
            (['img4.png'], ['img5.png'])  # Second split
        ]

        expected_train = ['img1.png', 'img2.png', 'img3.png']
        expected_val = ['img4.png']
        expected_test = ['img5.png']

        train_paths, test_paths, val_paths = split_data(img_paths, train_size=train_size, val_size=val_size, random_state=random_state)

        self.assertEqual(train_paths, expected_train)
        self.assertEqual(val_paths, expected_val)
        self.assertEqual(test_paths, expected_test)

        mock_train_test_split.assert_any_call(img_paths, train_size=train_size, random_state=random_state)
        mock_train_test_split.assert_any_call(['img4.png', 'img5.png'], test_size=val_size / (1 - train_size), random_state=random_state)
        mock_logger.info.assert_any_call("Starting data split with train_size=%s and val_size=%s", train_size, val_size)
        mock_logger.info.assert_any_call("Data split completed. Training set: %d, Testing set: %d, Validation set: %d", len(expected_train), len(expected_test), len(expected_val))





    @patch('process_data.os.makedirs')
    @patch('process_data.logger')
    def test_create_directories_with_set_types(self, mock_logger, mock_makedirs):
        base_dir = '/base/dir'
        set_types = ['train', 'test', 'val']
        mask_extensions = ['_mask1_mask.tif', '_mask2_mask.tif']

        create_directories(base_dir, set_types, mask_extensions)

        expected_calls = [
            (('/base/dir/train_images',), {'exist_ok': True}),
            (('/base/dir/train_masks',), {'exist_ok': True}),
            (('/base/dir/train_masks/mask1',), {'exist_ok': True}),
            (('/base/dir/train_masks/mask2',), {'exist_ok': True}),
            (('/base/dir/test_images',), {'exist_ok': True}),
            (('/base/dir/test_masks',), {'exist_ok': True}),
            (('/base/dir/test_masks/mask1',), {'exist_ok': True}),
            (('/base/dir/test_masks/mask2',), {'exist_ok': True}),
            (('/base/dir/val_images',), {'exist_ok': True}),
            (('/base/dir/val_masks',), {'exist_ok': True}),
            (('/base/dir/val_masks/mask1',), {'exist_ok': True}),
            (('/base/dir/val_masks/mask2',), {'exist_ok': True})
        ]

        actual_calls = mock_makedirs.call_args_list
        for call in expected_calls:
            self.assertIn(call, actual_calls)

        mock_logger.info.assert_any_call("Created directories: %s and %s", '/base/dir/train_images', '/base/dir/train_masks')
        mock_logger.info.assert_any_call("Created directories: %s and %s", '/base/dir/test_images', '/base/dir/test_masks')
        mock_logger.info.assert_any_call("Created directories: %s and %s", '/base/dir/val_images', '/base/dir/val_masks')

    @patch('process_data.os.makedirs')
    @patch('process_data.logger')
    def test_create_directories_without_set_types(self, mock_logger, mock_makedirs):
        base_dir = '/base/dir'
        set_types = []
        mask_extensions = ['_mask1_mask.tif', '_mask2_mask.tif']

        create_directories(base_dir, set_types, mask_extensions)

        expected_calls = [
            (('/base/dir/images',), {'exist_ok': True}),
            (('/base/dir/masks',), {'exist_ok': True}),
            (('/base/dir/masks/mask1',), {'exist_ok': True}),
            (('/base/dir/masks/mask2',), {'exist_ok': True})
        ]

        actual_calls = mock_makedirs.call_args_list
        for call in expected_calls:
            self.assertIn(call, actual_calls)

        mock_logger.info.assert_any_call("Created directories: %s and %s", '/base/dir/images', '/base/dir/masks')



    def test_crop_dimensions(self):
        y_min, y_max, x_min, x_max = crop_dimensions(self.img)
        # Allowing a margin of error due to morphological operations
        self.assertAlmostEqual(y_min, 30, delta=10)
        self.assertAlmostEqual(y_max, 470, delta=10)
        self.assertAlmostEqual(x_min, 30, delta=10)
        self.assertAlmostEqual(x_max, 470, delta=10)

    def test_padder(self):
        padded_img = padder(self.img, self.patch_size)
        step_size = int(self.patch_size / 8) * 7
        num_patches_h = (self.img.shape[0] + step_size - self.patch_size) // step_size + 1
        num_patches_w = (self.img.shape[1] + step_size - self.patch_size) // step_size + 1
        expected_shape = ((num_patches_h - 1) * step_size + self.patch_size,
                          (num_patches_w - 1) * step_size + self.patch_size)
        self.assertEqual(padded_img.shape[:2], expected_shape)

    @patch("os.makedirs")
    @patch("cv2.imwrite")
    def test_save_patches(self, mock_imwrite, mock_makedirs):
        img_patches = np.random.rand(4, self.patch_size, self.patch_size, 1).astype(np.float32)
        fpath = "raw/test_image.png"
        save_patches(fpath, img_patches)
        
        self.assertEqual(mock_imwrite.call_count, img_patches.shape[0])

    @patch('process_data.Dataset.File.upload_directory')
    @patch('process_data.DataPath')
    def test_upload_directory(self, mock_DataPath, mock_upload_directory):
        local_path = '/local/path'
        target_path = '/target/path'
        datastore = 'datastore_name'
        
        mock_dataset = MagicMock(spec=Dataset)
        mock_upload_directory.return_value = mock_dataset

        result = upload_directory(local_path, target_path, datastore)

        mock_DataPath.assert_called_once_with(datastore, target_path)
        mock_upload_directory.assert_called_once_with(
            src_dir=local_path,
            target=mock_DataPath(datastore, target_path),
            overwrite=True,
            show_progress=True
        )

        self.assertEqual(result, mock_dataset)


    @patch('process_data.save_patches')
    @patch('process_data.patchify')
    @patch('process_data.padder')
    @patch('process_data.crop_dimensions')
    @patch('process_data.cv2.imread')
    @patch('process_data.os.listdir')
    def test_process_images(self, mock_listdir, mock_imread, mock_crop_dimensions, mock_padder, mock_patchify, mock_save_patches):
        # Set up mock return values
        mock_listdir.return_value = ['mask1_mask.tif', 'mask2_mask.tif']

        # Create a mock image array
        mock_image = np.ones((100, 100), dtype=np.uint8)
        mock_imread.return_value = mock_image

        # Set up other mock return values
        mock_crop_dimensions.return_value = (0, 50, 0, 50)
        mock_padder.return_value = np.ones((128, 128), dtype=np.uint8)
        mock_patchify.return_value = np.ones((4, 4, 256, 256), dtype=np.uint8)

        set_type = 'train'
        set_paths = ['path/to/image1.png', 'path/to/image2.png']
        img_patched_dir = 'path/to/patched'
        img_dir = 'path/to/images'
        selected_masks = ['_mask.tif']
        patch_size = 256
        mode = 'a'

        process_images(set_type, set_paths, img_patched_dir, img_dir, selected_masks, patch_size, mode)

        # Assertions to ensure the image processing pipeline is executed
        self.assertEqual(mock_imread.call_count, 2)  # Ensure imread is called twice (for two images)
        self.assertEqual(mock_crop_dimensions.call_count, 2)  # Ensure crop_dimensions is called twice
        self.assertEqual(mock_padder.call_count, 2)  # Ensure padder is called twice
        self.assertEqual(mock_patchify.call_count, 2)  # Ensure patchify is called twice

        # Ensure save_patches is called with the correct arguments
        expected_calls = [call(f'{img_patched_dir}/train_images/image1.png', mock_patchify.return_value),
                          call(f'{img_patched_dir}/train_images/image2.png', mock_patchify.return_value)]
        mock_save_patches.assert_has_calls(expected_calls, any_order=True)

    @patch('process_data.upload_directory')
    @patch('process_data.process_images')
    @patch('process_data.create_directories')
    @patch('process_data.split_data')
    @patch('process_data.validate_image_paths')
    @patch('process_data.select_masks')
    @patch('process_data.get_image_paths_and_masks')
    @patch('process_data.check_directory_and_handle_contents')
    @patch('process_data.select_directory')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('process_data.logger')
    @patch('os.path.abspath')
    @patch('os.path.join')
    def test_main(self, mock_join, mock_abspath, mock_logger, mock_parse_args, mock_select_directory, 
                  mock_check_directory_and_handle_contents, mock_get_image_paths_and_masks, mock_select_masks,
                  mock_validate_image_paths, mock_split_data, mock_create_directories, mock_process_images, 
                  mock_upload_directory):
        # Set up the mock arguments
        mock_parse_args.return_value = argparse.Namespace(mode='default', store='local', dir=None)
        mock_select_directory.return_value = 'chosen_directory'
        mock_check_directory_and_handle_contents.return_value = (True, 'default', 'img_patched_dir')
        mock_get_image_paths_and_masks.return_value = (['img_path'], ['_mask.tif'])
        mock_select_masks.return_value = ['_mask.tif']
        mock_validate_image_paths.return_value = ['img_path']
        mock_split_data.return_value = (['train_img_path'], ['test_img_path'], ['val_img_path'])
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_abspath.return_value = 'absolute_path'

        # Capture the output during the main function execution
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            main()
        
        # Ensure the appropriate functions were called with correct arguments
        mock_select_directory.assert_called_once_with('data/raw', mode='default')
        mock_check_directory_and_handle_contents.assert_called_once_with('chosen_directory', mode='default', store='local')
        mock_get_image_paths_and_masks.assert_called_once_with('chosen_directory/images/', mode='default')
        mock_select_masks.assert_called_once_with(['_mask.tif'], mode='default')
        mock_validate_image_paths.assert_called_once_with(['img_path'], ['_mask.tif'], 'chosen_directory/images/', mode='default')
        mock_split_data.assert_called_once_with(['img_path'])
        mock_create_directories.assert_called_once_with('img_patched_dir', ['train', 'test', 'val'], ['_mask.tif'])
        mock_process_images.assert_any_call('train', ['train_img_path'], 'img_patched_dir', 'chosen_directory/images/', ['_mask.tif'], 256, 'default')
        mock_process_images.assert_any_call('test', ['test_img_path'], 'img_patched_dir', 'chosen_directory/images/', ['_mask.tif'], 256, 'default')
        mock_process_images.assert_any_call('val', ['val_img_path'], 'img_patched_dir', 'chosen_directory/images/', ['_mask.tif'], 256, 'default')

if __name__ == '__main__':
    unittest.main()
