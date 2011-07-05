The <b>memex browser</b> is a javascript/html based visualization of
all of the exemplar detection results.

It provides a sort of graph-based traversal of a large collection of
images.  Since the drawing of detection boxes and associations is done
in javascript, there is no need to generate images with painted in
boxes.  This means that for a single image, with 1000s of detection
results, there is no need to waste disk space.  Instead, the work is
shifted to the web-server and the client-side boundinb box drawing.