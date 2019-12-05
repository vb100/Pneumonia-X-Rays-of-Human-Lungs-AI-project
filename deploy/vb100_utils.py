#-------------------------------------------------------------------------------#
# Module dedicated to Pneumonia Detection on Keras Model                        #
# Part: Main functions and Procedures                                           #
# Date of preparation: 2019 07 16–26                                            #
# © Vytautas Bielinskas                                                         #
#-------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------#
# Import modules and packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
import time, datetime, os, stat, glob, shutil, itertools
import itertools
from pylab import rcParams
from os.path import dirname as up
from PIL import Image
from shutil import copyfile
from datetime import date


#-------------------------------------------------------------------------------#
# Plot Raw Image Data
def plots(ims, figsize=(14, 7.5), rows=1, interp=False, titles=None):
    '''
    Args:	
        -- ims : given list of images (list)
        -- figsize : wanted size of plotting images (width and height)
        -- rows : wanted rows to be ploted per one request
    '''

    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=10)
        plt.imshow(ims[i].squeeze(), interpolation=None if interp else 'none')


#-------------------------------------------------------------------------------#
# If Any Issues caused by non-readable directories
def on_rm_error(func, path, exc_info):
    '''
    Given path contains the path of the file that couldn't be removed
    let's just assume that it's read-only and unlink it.
    '''

    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)

    return None


#-------------------------------------------------------------------------------#
# Delete files in a single folder
def delete_files_in_folder(folder_path):
    '''
    Args:
        -- folder_path : a given URL to folder contains any content
    '''

    os.chmod(folder_path, 0o777)
    print('given folder address: {}'.format(folder_path))
    shutil.rmtree(folder_path, onerror = on_rm_error)
    print('folder deleted.')

    return None


#-------------------------------------------------------------------------------#
# Rename folder by a given folder name in given full URLs
def rename_folder_with_single_class(current_folder_path, renamed_folder_path):
    '''
        -- folder_path : a given URL to folder contains any content    
    '''

    os.rename(current_folder_path, renamed_folder_path)
    print('Directory for single class has been changed from\
    {} to {}'.format(current_folder_path, renamed_folder_path))
    
    return None


#-------------------------------------------------------------------------------#
# Check if Abstract class folder is in any project subdirectory
def abstract_class_exists(ABSTRACT_CLASS, l_DIRS):
	'''
		-- ABSTRACT_CLASS : name of abstract class which must be splited after.
		-- l_DIRS : list of train, test and val. sets for the project.
	'''

	l_subs = []
	for folder in l_DIRS:
	    subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]

	    l_subs.append(subfolders)
	# Concat separate list of folders (list of lists) to single vector
	l_subs = [folder for folders in l_subs for folder in folders]

	response = ABSTRACT_CLASS in l_subs

	if response:
		print('Performing raw data restructuring. Please wait.\n')
	else:
		print('Data are valid and ready to be feeded to the model already.\n')

	return response


#-------------------------------------------------------------------------------#
# Get list of classes for each set
def classes_for_each_set(l_DIRS):
	'''
		-- l_DIRS : list of train, test and val. sets for the project.
	'''

	d = {} # Reserved space for a single list of classes within set (dictionary)
	for folder in l_DIRS:
	    subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]

	    if 'train'.upper() in folder.upper():
	    	d['TRAIN'] = subfolders

	    elif 'test'.upper() in folder.upper():
	    	d['TEST'] = subfolders

	    else:
	    	d['VALIDATION'] = subfolders

	return d


#-------------------------------------------------------------------------------#
# Original data structuring for specific schema
def structure_origin_data(l_DIRS, IMAGE_FORMAT, POSITIVE_CLASS):
    '''
        -- l_DIRS : list of directories for training, test and val. sets.
        -- IMAGE_FORMAT : format of files to be readed.
        -- POSITIVE_CLASS : name of Positive class in the task.
    '''

    cwd = os.getcwd() # Get Current Working Directory
    print('Current Working Directory: {}'.format(cwd))

    dims_w = []
    dims_h = []

    dirs_to_delete = []

    for this_dir in l_DIRS:
        print('\n*** Handling workspace: {} ***'.format(this_dir))
        # Keep Temporary Directory
        temp_dir = '{}/{}'.format(cwd, this_dir) 
        os.chdir(temp_dir)

        # Get list of Folders in Temporary Directory
        list_folders = [f.path for f in os.scandir(temp_dir) if f.is_dir()]
        print('| Folders found:\n|| {}'.format(list(list_folders)))
        
        if len(list_folders) > 0:

            # Got through subfolders
            for this_folder in list_folders:
                #1. Change CWD to Subfolder
                os.chdir(this_folder)
                print('\n| Entering to: {}'.format(this_folder))
                
                classes = [] # Classes in this subfolder
                files = []   # Filelist in this Subfolder with a set Class
                
                #2. Scanning image data
                for image_file in glob.glob('*.{}'.format(IMAGE_FORMAT)):
                    
                    d = {} # Temporary Data for a Single Image in Subfolder
                    
                    #3. Open Image to Extract Data
                    with Image.open(image_file) as opened_image:    # Open Image file
                        opened_image_dims = opened_image.size       # Extract Dimmensions
                        opened_image.close()                        # Close Image File
                    
                    #4. Identify the Class (label)
                    if '_' in image_file:
                        this_class = image_file.upper().split('_')[1].split('_')[0].replace(' ', '')
                    else:
                        this_class = POSITIVE_CLASS

                    classes.append(this_class)
                    
                    #5. Assign file to the Class
                    d['Class'] = this_class
                    d['Filename'] = image_file
                    files.append(dict(d))
                    
                    #6. Add Dimensions to lists
                    dims_w.append(opened_image_dims[0])
                    dims_h.append(opened_image_dims[1])
                    
                #7. Get Unique Classes in Scanned Subfolder
                unique_classes = list(set(classes))
                files_df = pd.DataFrame(files)

                print('||| Unique Classes Found: {}'.format(unique_classes))
                print('||| Files found in total: {}'.format(len(files)))

                for this_class, i in zip(unique_classes, range(0, len(unique_classes))):

                    print('--> {} = {}'.format(this_class, len(files_df[files_df['Class'] == this_class])))
                    dir_for_class = '{}/{}'.format(temp_dir, this_class)

                    #8. Generate List of Files to be Moved
                    filelist = list(files_df[files_df['Class']==this_class]['Filename'])
                    print('|||--> Filelist for {} is created.'.format(this_class))

                    # Make a Copy in Level Up Folder
                    #9. Create a Special Folder for A Specifed Class in Parent Folder
                    os.chdir(temp_dir)
                    
                    if len(unique_classes) > 1:
                        try:  

                            os.mkdir(dir_for_class)
                            print('|||| Successfully created the directory for {} on {}'.format(this_class, dir_for_class))

                        except OSError:

                            print('!!!! Creation of the directory {} failed.'.format(dir_for_class))

                        os.chdir(this_folder)
                        for f in filelist:
                            shutil.copy(f, dir_for_class)

                        #11. Add Current Folder to the list for Deleting
                        dirs_to_delete.append(this_folder)
                        
                    elif (len(unique_classes) == 1) and (unique_classes[0].upper() != POSITIVE_CLASS.upper()):
                        dir_for_class = '{}/{}'.format(temp_dir, unique_classes[0].upper())
                        rename_folder_with_single_class(this_folder, dir_for_class)
                                      
    #12. Deleting Used Directories
    time.sleep(5)
    os.chdir(cwd)
    for deleting in set(dirs_to_delete):

        print('\n--> Deleting Folder: {}'.format(deleting))
        delete_files_in_folder(deleting)
            
    print('*** All Used Folders have been removed from the system. ***')
                
    # Averaging Dimmensions of Scanned Images
    avg_dims_w = int(np.average(dims_w))
    avg_dims_h = int(np.average(dims_h))
    print('\n: Average Image Width = {}'.format(avg_dims_w))
    print(': Average Image Height = {}'.format(avg_dims_h))

    return None


#-------------------------------------------------------------------------------#
# Plot Result Graph for Accuracy, Loss and Validation
def plot_model_result(model):
	'''
		-- model : Keras model.
	'''

	rcParams['figure.figsize'] = 14, 4 # Set plot size

	# Plot #1

	y1 = model.history.history['val_acc']
	y2 = model.history.history['acc']

	_ = plt.title('Model Results', family='Arial', fontsize=12)

	_ = plt.plot(y1, 
		color='blue', linewidth=1.5, marker='D', markersize=5,
		label='Validation acc.')
	_ = plt.plot(y2, 
		color='#9999FF', linewidth=1.5, marker='D', markersize=5,
		label='Training acc.')

	_ = plt.xlabel('Epochs', family='Arial', fontsize=10)
	_ = plt.ylabel('Score', family='Arial', fontsize=10)

	_ = plt.yticks(np.arange(0., 1.25, 0.1),
				   family='Arial', fontsize=10)

	if len(model.history.history['acc']) < 51:
		_ = plt.xticks(np.arange(0, len(model.history.history['acc']), 1),
					   family='Arial', fontsize=10)

	_ = plt.ylim((0., 1.))

	_ = plt.fill_between(np.arange(0, len(model.history.history['acc']), 1),
						 model.history.history['acc'], 0,
						 color = '#cccccc', alpha=0.5)

	_ = plt.grid(which='major', color='#cccccc', linewidth=0.5)
	_ = plt.legend(loc='best', shadow=True)
	_ = plt.margins(0.02)

	_ = plt.show()

	# Plot #2
	_ = plt.clf()

	_ = plt.plot(model.history.history['val_loss'], 
		color='red', linewidth=1.5, marker='D', markersize=5,
		label='Validation loss')
	_ = plt.plot(model.history.history['loss'], 
		color='#FF7F7F', linewidth=1.5, marker='D', markersize=5,
		label='Loss')

	_ = plt.xlabel('Epochs', family='Arial', fontsize=10)
	_ = plt.ylabel('Loss score', family='Arial', fontsize=10)

	if len(model.history.history['acc']) < 51:
		_ = plt.xticks(np.arange(0, len(model.history.history['acc']), 1),
					   family='Arial', fontsize=10)
	_ = plt.yticks(family='Arial', fontsize=10)

	_ = plt.grid(which='major', color='#cccccc', linewidth=0.5)
	_ = plt.legend(loc='best', shadow=True)
	_ = plt.margins(0.02)

	_ = plt.show()

	return None


#-------------------------------------------------------------------------------#
# Convert RGB Images to Grayscale Images
def rgb_to_grayscale(imgs_set):
	'''
		-- imgs_set : set of RGB images (3 channels)
	'''

	return tf.image.rgb_to_grayscale(imgs_set, name=None)


#-------------------------------------------------------------------------------#
# Save Model Result to Pandas DataFrame and then to CSV file
def save_model_result(model):
	'''
		-- model : compiled model on the given data.
	'''

	# Extract model result data to separate data vectors
	data_val_acc = list(model.history.history['val_acc'])
	data_acc = list(model.history.history['acc'])
	data_val_loss = list(model.history.history['val_loss'])
	data_loss = list(model.history.history['loss'])

	# Convert Model result data to dataframe by using dictionary dat structure
	d = {}
	d['val_acc'] = data_val_acc
	d['acc'] = data_acc
	d['val_loss'] = data_val_loss
	d['loss'] = data_loss

	df = pd.DataFrame(d)
	print(df)

	# Get a current timestamp
	timestamp = str(datetime.datetime.now()).replace(":","-")[:-10].replace(' ', '_')
	filename = 'model_results_{}.csv'.format(timestamp)
	df.to_csv(filename, encoding='utf-8')

	print('\n\nResult data is saved as file: {}'.format(filename))

	return df