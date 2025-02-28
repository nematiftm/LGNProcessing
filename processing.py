import cv2
import os
import glob
from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA

class Input:
    def __init__(self):
        self.input_type      = 'image'          # Type of input: image or video or mat
        self.dt              = 1                # Time interval for videos
        self.duration        = []               # Total duration for videos
        self.channel         = None             # channel =1 grayscale, =3 color frames

        self.nframes         = []               # Number of images or video frames
        self.nvideos         = None             # Number of video files
        self.npatch          = None             # Number of patches

        self.size            = None             # Frame size
        self.patch_size      = None             # Patch size
        self.RF_size         = None             # Receptive Field size
        self.overlap         = None             # Overlap Size of Receptive fields
        
        self.input_mat       = []               # Placeholder for storing the input matrix
        self.proc_input      = []               # Placeholder for storing the processed input data
        self.patches_list    = []               # Placeholder for storing the patch data
        self.RF_patches      = []               # Placeholder for storing the receptive field patch
      
    def input_read(self, input_type, file_format, input_directory=None, data_type=None, size=512, channel=1, dt=1e-3, data=None):
        self.input_type = input_type
        self.size       = size
        self.dt         = dt
        self.channel    = channel

        if input_directory:
            if os.path.isdir(input_directory): 
                input_list         = glob.glob(os.path.join(input_directory, "*" + file_format))
                if self.input_type == "image":
                    self.nframes    = [len(input_list)]
                    self.nvideos    = 0

                elif self.input_type == "mat":
                    tmp  = loadmat(input_list[0])
                    if data_type == 'raw':
                        data = tmp['IMAGESr']

                    elif data_type== 'processed':
                        data = tmp['IMAGES_WHITENED']
                    
                    else:
                        raise FileNotFoundError("Image directory does not exist.")
                    
                    self.nframes   = [np.shape(data)[-1]]
                    self.nvideos    = 0

                self.input_data  = []
                
            else:
                raise FileNotFoundError("Image directory does not exist.")
            
        elif self.input_type == "array":
            self.input_data  = data
            self.nframes     = [len(data)]
            self.nvideos     = 0
            
        #Image Input
        if self.input_type == "image":
            for i, file_path in enumerate(input_list):
                img     = cv2.imread(file_path)
                w, h,_   = img.shape
                if w==h and h==size:
                    resized_image = img
                else:
                    resized_image = cv2.resize(img, [size, size])
                if self.channel == 1:
                    img_gray  = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray  = resized_image
                self.input_mat.append(img_gray)

        elif self.input_type =='mat':
            for i in range(self.nframes[0]):
                img_bgr       = data[:,:, i]
                w, h          = img_bgr.shape
                if w==h and h==size:
                    resized_image = img_bgr
                else:
                    resized_image = cv2.resize(img_bgr, [size, size])
                self.input_mat.append(resized_image)
        
        elif self.input_type =='video':
            for i, file_path in enumerate(input_list):
                self.nvideos        = len(input_list)
                video_capture       = cv2.VideoCapture(file_path)
                # Get the total number of frames
                total_frames        = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                # Get the original frame rate of the video
                original_frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
                # Calculate the frame skip interval based on the desired and original frame rates
                frame_skip_interval = int(original_frame_rate * self.dt)
                # Calculate the duration in seconds
                duration = total_frames / original_frame_rate
                self.duration.append(int(duration))
                frame_count         = 0
                nsample             = 0
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    # Resize the frame
                    w, h,_          = frame.shape
                    if w==h and h==size:
                        resized_frame = frame
                    else:
                        resized_frame = cv2.resize(frame, [size, size])
                    # Skip frames based on the frame skip interval
                    if frame_count % frame_skip_interval == 0:
                        frame_count += 1
                        if channel == 1:
                            frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        else:
                            frame_gray = resized_frame
                        nsample += 1
                        self.input_mat.append(frame_gray)                    
                    else:
                        frame_count += 1
                self.nframes.append(nsample)
                video_capture.release()
                
        elif self.input_type == "array":
            self.input_mat = np.copy(self.input_data)
            
        else:
            raise ValueError("Invalid input type.")
        
        self.input_mat = np.array(self.input_mat)

    def process_fn(self, image, kernel_params):
        type = kernel_params['type']

        if self.channel == 1:
            processed_image = np.zeros([self.size, self.size])
        else:
            processed_image = np.zeros([self.size, self.size, self.channel])

        if type == 'DOG':
            for i in range(self.channel):
                if self.channel == 1:
                    g1 = cv2.GaussianBlur(image, kernel_params['size'], kernel_params['sigma'][0])
                    g2 = cv2.GaussianBlur(image, kernel_params['size'], kernel_params['sigma'][1])
                    processed_image = g1 - kernel_params['gamma']*g2
                else:
                    g1 = cv2.GaussianBlur(image[:, :, i], kernel_params['size'], kernel_params['sigma'][0])
                    g2 = cv2.GaussianBlur(image[:, :, i], kernel_params['size'], kernel_params['sigma'][1])
                    processed_image[:, :, i] = g1 - kernel_params['gamma']*g2

        elif type == 'Whiten':
            
            N               = self.size
            fx, fy          = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2))
            rho             = np.sqrt(fx**2 + fy**2)
            f_0             = kernel_params['f_0']
            filt            = rho * np.exp(-((rho / f_0)**kernel_params['n']))
            
            for i in range(self.channel):
                if self.channel == 1:
                    If               = np.fft.fftshift(np.fft.fft2(image))
                    filtered_fft     = If * filt
                    processed_image  = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
                    processed_image  *= np.sqrt(kernel_params['scale']/np.var(processed_image)) 
                else:
                    If                        = np.fft.fftshift(np.fft.fft2(image[:, :, i]))
                    filtered_fft              = If * filt
                    processed_image[:, :, i]  = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
                    processed_image[:, :, i]  *= np.sqrt(kernel_params['scale']/np.var(processed_image)) 
        else:
            raise FileNotFoundError("Processing type does not exist.")        
        return processed_image

    def processing(self, params = {}, save_directory=None):
        if params['status']:
            if params['zero_mean']:
                feature_means  = self.input_mat.mean(axis=0)
                data           = self.input_mat - feature_means
            else:
                data = self.input_mat

            for i in range(sum(self.nframes)):
                img_bgr       = data[i]

                if params['status']:
                    processed_img = self.process_fn(img_bgr, params)
                else:
                    processed_img = img_bgr 
                
                if save_directory is not None:
                    os.makedirs(save_directory, exist_ok=True)
                    if processed_img.dtype != np.uint8:
                        # Find the minimum and maximum values in the image
                        min_val = np.min(processed_img)
                        max_val = np.max(processed_img)
                    
                        # Scale the image data to 0-255
                        # This line handles cases where all values are the same (max_val == min_val)
                        if max_val > min_val:
                            tmp = (processed_img - min_val) / (max_val - min_val) * 255
                        else:
                            tmp = np.zeros(processed_img.shape, dtype=np.uint8)

                        # Convert to uint8
                        tmp = tmp.astype(np.uint8)
                    else:
                        tmp = processed_img

                    cv2.imwrite(save_directory + "{0:03d}.jpg".format(i + 1), tmp)

                self.proc_input.append(processed_img)

            self.proc_input = np.array(self.proc_input)
        else:
            self.proc_input = np.copy(self.input_mat)

    def create_patch(self, patch_size):   
        self.patch_size   = patch_size
        self.patches_list = []
        for i in range(sum(self.nframes)):
            if len(self.proc_input) !=0:
                image = self.proc_input[i]
            else:
                image = self.input_mat[i]
            patches = []
            for y in range(0, self.size-patch_size[0]+1, patch_size[0]):
                for x in range(0, self.size-patch_size[1]+1, patch_size[1]):
                    if self.channel == 3:
                        patch = image[y:y+patch_size[0], x:x+patch_size[1], :]
                    else:
                        patch = image[y:y+patch_size[0], x:x+patch_size[1]]
                    patches.append(patch)
            self.patches_list.append(patches)
        self.patches_list = np.array(self.patches_list)
        self.npatch       = np.shape(self.patches_list)[1]

    def create_RFpatch(self, RF_size, overlap_size, process_flg=False):
        self.RF_size     = RF_size
        self.RF_patches  = []
        h_step           = RF_size[0] - overlap_size[0] - 1
        w_step           = RF_size[1]  - overlap_size[1] -1 

        for i in range(np.shape(self.patches_list)[0]):
            patches   = []
            for j in range(np.shape(self.patches_list)[1]):
                patch       = self.patches_list[i,j]
                RF_patches_y = []
                for y in range(0, self.patch_size[0] - RF_size[0] + 1, h_step):
                    RF_patches_x = []
                    for x in range(0, self.patch_size[1] - RF_size[1] + 1, w_step):
                        if self.channel == 3:
                            RF_patch = patch[y:y+RF_size[0], x:x+RF_size[1], :]
                        else:
                            RF_patch = patch[y:y+RF_size[0], x:x+RF_size[1]]
                            if process_flg:
                                RF_patch = np.multiply(RF_patch, GaussianMask(RF_size=self.RF_size))
                                RF_patch = (RF_patch - np.mean(RF_patch))
                        RF_patches_x.append(RF_patch)
                    RF_patches_y.append(RF_patches_x)
                patches.append(RF_patches_y)
            self.RF_patches.append(patches)
        self.RF_patches = np.array(self.RF_patches)

    def retina_onoff(self, flatten_flg):
        """
        Separate the RF patches into ON and OFF arrays. Flatten the last two dimensions if flatten_flg is True.
        :param flatten_flg: Boolean flag to indicate whether to flatten the last two dimensions.
        :return: Tuple of (ON_array, OFF_array) with flattened or original shape.
        """
        # Create ON and OFF arrays based on the RF patches
        # axes_to_average = tuple(range(self.RF_patches.ndim - 3))
        # zmean_Array     = self.RF_patches - self.RF_patches.mean(axis=axes_to_average)
        ON_array  = np.where(self.RF_patches > 0, self.RF_patches, 0)
        OFF_array = np.where(self.RF_patches < 0, -self.RF_patches, 0)

        if flatten_flg:
            # Flatten only the last two dimensions
            last_two_size  = ON_array.shape[-2] * ON_array.shape[-1]
            first_two_size = ON_array.shape[0] * ON_array.shape[1]
            ON_flat        = ON_array.reshape(*ON_array.shape[:-2], last_two_size)
            ON_flat        = ON_flat.reshape(first_two_size, *ON_flat.shape[2:])
            OFF_flat       = OFF_array.reshape(*OFF_array.shape[:-2], last_two_size)
            OFF_flat       = OFF_flat.reshape(first_two_size, *OFF_flat.shape[2:])
            return ON_flat, OFF_flat
        else:
            # Return the original arrays if no flattening is required
            return ON_array, OFF_array
        
def GaussianMask(RF_size, sigma_val=5):
    x = np.arange(0, RF_size[0], 1, float)
    y = np.arange(0, RF_size[1], 1, float)
    x, y = np.meshgrid(x,y)

    x0 = RF_size[0] // 2
    y0 = RF_size[1] // 2
    mask = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*(sigma_val**2)))
    return mask / np.sum(mask)