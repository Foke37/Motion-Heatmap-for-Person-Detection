import numpy as np
from time import time
import cv2


class Heatmap(object):
    """
    This class takes YOLO detections from a camera, translates to reduced working space, 
    which allows less computing and better generalization (e.g. 1920x1080 -> 640x480). After that,
    those coordinates are saved and kept until a thresholded number of frames is hit, which leds to
    averaging of the saved detections and, posteriorly, construction of the heatmap with
    computed averages.
    """

    def __init__(self, camera_dims):
        """
        Heatmap class constructor

        Parameters
        ----------
            `camera_dims` : tuple
                Two values corresponding respectively to camera width and height

        Returns
        -------
            None
        """
        # Declare global variables
        # self.p_count = 0 # Person count
        camera_dims = 640, 480
        self.camera_width = camera_dims[0]
        self.camera_height = camera_dims[1]
       
        self.grid_side = max(camera_dims)//32 # grid_side

        self.dets_thresh = 0
        self.last_dets = None

        self.grids_remove_movement_temp = []
        self.grids_remove_movement = np.zeros((self.grid_side, self.grid_side), dtype=np.uint16)

        colorspace_bar = np.array([np.linspace(255,0,self.camera_height,dtype=np.uint8) for _ in range(15)])
        self.colorspace_bar = cv2.applyColorMap(colorspace_bar.transpose(), cv2.COLORMAP_JET)
        self.white_margin = 255*np.ones((self.camera_height,8,3),dtype=np.uint8)

    def _grid_averages(self):
        """
        Average the number of detections from a set.(e.g. takes the average number of people in scene 
        every 3 detections).

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        new_grid = np.zeros((self.grid_side, self.grid_side), dtype=np.uint8)

        temp_grids = self.grids_remove_movement_temp

        for temp_grid in temp_grids:
            new_grid += temp_grid
        new_grid = (new_grid // len(temp_grids))
        self.grids_remove_movement_temp = []

        self.grids_remove_movement += new_grid
        self.grids_remove_movement = np.clip(self.grids_remove_movement, 0, 255)


    def _box_center(self, xyxy):
        """
        Convert bounding box coordinates from top-right/bottom-left to center. (XlYt,XrYb)->(XcYc)

        Parameters
        ----------
            `xyxy`: tuple
                Tuple containing bounding box top-right/bottom-left coordinates for conversion

        Returns
        -------
            `center`: tuple
                Tuple containing Xc,Yc coordinates converted from XlYt,YtYb
        """

        center = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2.0), int(xyxy[1]+(xyxy[3]-xyxy[1])/2.0))
        return center

    
    def store_detections(self, detections ):
        """
        Store detections for future averaging.
    
        Parameters
        ----------
            `detections`: list
                Python list containing bounding box coordinates, confidence and class.
    
        Returns
        -------
            None
        """

        self.dets_thresh += 1
        temp_grid = np.zeros((self.grid_side, self.grid_side), dtype = np.uint8)
        
        self.last_dets = []
        if detections is not None and len(detections):
            for *xyxy, _, _ in reversed(detections):
                x, y = self._box_center(xyxy)
                if x >= self.camera_width or y >= self.camera_height:
                    continue

                x_norm = x/float(self.camera_width)  # Coord in range:
                y_norm = y/float(self.camera_height) #     [0,1]

                x_grid = int((x_norm * self.grid_side) // 1) 
                y_grid = int((y_norm * self.grid_side) // 1) 

                temp_grid[y_grid][x_grid] += 1                                                                                                                                                    

                self.last_dets.append((y, x))
        self.grids_remove_movement_temp.append(temp_grid)

        if self.dets_thresh >= 5:
            self._grid_averages()   

    def sliding_window(self,image, stepSize, windowSize):
        value = 150
        x_out , y_out = 0,0
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                # cv2.rectangle(image, (x, y), (x + windowSize[0], y + windowSize[1]), 0, 2)
                k = image[y:y + windowSize[1], x:x + windowSize[0]].sum()
                if k > value:
                    value = k
                    x_out = x
                    y_out = y
                    
        return x_out,y_out
    
    def Euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return  dist 

    def _generate_heatmap(self , VDOdetect):
        """
        Draw averaged detections on heatmap.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        z = (self.grids_remove_movement).astype(np.uint8)
        
        z = cv2.resize(z, (self.camera_width, self.camera_height))
        z = cv2.GaussianBlur(z, (5,5), 0)
        im0 = VDOdetect  
        image = z
        wX,wY = 5,5
        r = 35
        x_out,y_out = self.sliding_window(z, stepSize=6, windowSize=(wX, wY))
        people_point = []
        red_zone_list = []
        
     
        image = cv2.applyColorMap(z, cv2.COLORMAP_JET)
        # result_overlay = cv2.addWeighted(im0, 0.5, image, 0.5, 0)
        
        if self.last_dets:
            for det in self.last_dets:
                y, x = det
                people_point.append((x,y))                
                cv2.circle(image, (x,y),5, (0,255,0), -1)
                cv2.circle(image, (x_out, y_out), r, (86, 255, 255), 2)
                
                # # make diameter line of circle
                # cv2.line(image, (x_out,y_out-30),(x_out,y_out+30), (0,0,255), 2)

                # cv2.circle(image, (x_out, y_out), 5, (0, 0, 0), -1)
                
                # cv2.rectangle(image, (x_out-30,y_out-30),(x_out+30,y_out+30), (0,0,255), 2)
                # result_overlay = cv2.addWeighted(im0, 0.5, image, 0.5, 0)
            
            for i in range(len(people_point)):
                distance = self.Euclidean_distance((x_out, y_out), people_point[i])
                if distance < r:
                    red_zone_list.append(people_point[i])
                    cv2.circle(image, people_point[i], 5, (0, 0, 255), -1)
                # if  35 < distance < 37:
                #     self.p_count += 1

        
        
        # text2 = "All People Counted: %s" % str(self.p_count) 			# Count People at Risk
        # location2 = (10,25)												# Set the location of the displayed text
        # cv2.putText(image, text2, location2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,186,186), 2, cv2.LINE_AA)  # Display Text      

        text = "People at Redzone: %s" % str(len(red_zone_list)) 			# Count People at Risk
        location = (10,55)												# Set the location of the displayed text
        cv2.putText(image, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,186,186), 2, cv2.LINE_AA)  # Display Text

        
        # margin = self.white_margin
        # image = np.hstack((margin, image, margin, self.colorspace_bar, margin))



        # text 
        # width = self.camera_width
        # height = self.camera_height
        # text = f'Width: {width} Height: {height}'
        # location = (10,50)
        # cv2.putText(image, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # #result_overlay
        # cap = cv2.VideoCapture("vtest_7krCffnz.mp4")
        # ret, frame = cap.read() 
        # result_overlay = cv2.addWeighted(frame, 0.3, image, 0.7, 0)

        result_overlay = cv2.addWeighted(im0, 0.3, image, 0.7, 0)
                         

        cv2.imshow('heatmap', image)
        cv2.imshow('result_overlay', result_overlay)
        cv2.imshow('z', z  )
        
        # cv2.imshow('VDOdetect', z)
        print("people at redzone: ", len(red_zone_list))


        # Save results
        cv2.imwrite('result_overlay.jpg', result_overlay)
        cv2.imwrite('heatmap.jpg', image)
        #save video
        length = 795
        
        
        
            

        
    

        
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit(0)





