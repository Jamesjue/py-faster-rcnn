
### Basic Feature

#### Detect image and display

Basic Procedure
* User upload image
* Detect
* Display the image

Requirement
* Infrequent invocation
* Return the detected image



#### Detect image and respond bounding box 

Basic Procedure
* User upload image
* Detect
* Return the bounding box 

Requirement
* Frequent invocation
* Direct access from the user and the container application
* Return bounding box



### Design

Single Interface: detect
> Input: image, confidence, format  
> Output: bounding box or image 

For each kind of invocation
* First check if container is running
* If running, redirect to the interface 
* If not running, start the container, redirect to the interface 

Interface specify:
* detect(image=[image file the POST form], confidence=[float, the minimal threshold for filtering boxes], format=['img', 'box'])
* return: image=[image file in the response], box=[json format of detected box and score]

