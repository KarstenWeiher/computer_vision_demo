import time
import matplotlib.pyplot as plt
import numpy as np
from skimage import segmentation
from skimage import draw

def iterative_snake(img_cut, img_blur, n_iter, **kwargs):
    
    shape = img_cut.shape
    rr, cc = draw.rectangle_perimeter((0, 0), end = (shape[0], shape[1]))
    init = np.array([rr, cc]).T
    init = init[::2,:]
    snake = init
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img_cut, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    line1, = ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    
    for i in range(n_iter):
        snake = segmentation.active_contour(img_blur, snake, **kwargs)
        
        # updating data values
        line1.set_xdata(snake[:, 1])
        line1.set_ydata(snake[:, 0])

        # drawing updated values
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.5) # Sleep for 3 seconds
    
    return snake   