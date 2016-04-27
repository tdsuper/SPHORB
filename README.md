# SPHORB



----------------
What is SPHORB?
----------------

The SPHORB (Spherical ORB) package is an implementation in OpenCV of the algorithm 
introduced in "SPHORB: A Fast and Robust Binary Feature on the Sphere" by Zhao et al.
This algorithm is designed to detect and describe the features for spherical panoramic 
images, which are more and more easily obtained for common users. Based on a nearly 
regular hexagonal grid parametrization of the sphere - geodesic grid, we can adopt the 
planar ORB features to the spherical domain and achieve satisfactory performance.


-----------------
Conditions of use
-----------------

SPHORB is distributed under the GNU General Public License.  For information on 
commercial licensing, please contact the authors at the contact address below.

If you use this package in published work, please cite our work as
@article{zhao-SPHORB,
    author   = {Qiang Zhao and Wei Feng and Liang Wan and Jiawan Zhang},
    title    = {SPHORB: A Fast and Robust Binary Feature on the Sphere},
    journal  = {International Journal of Computer Vision},
    doi      = {10.1007/s11263-014-0787-4},
    year     = {2015},
    volume   = {113},
    number   = {2},
    pages    = {143-159},
}


---------------
What's included
---------------

Before using SPHORB, you need to install the OpenCV library. 
OpenCV 2.4.2 is used in our implementation.

In SPHORB.rar, there are some folders and files.
    -- Data folder
                    the data used to accelerate or simplify the algorithm

    -- Image folder
                    the first image pair is for camera rotation with the source image from SUN360 database[1], 
		    the second pair is for camera movement with the two images from Google Street View (C).

    -- pfm.h pfm.cpp
                    reader for PFM(Portable Float Map) file

    -- utility.h utility.cpp
                    the utility functions for ratio matching strategy and drawing matches
		    (different with the "drawMatches" function of OpenCV)

    -- detector.h detector.cpp nonmax.cpp
                    spherical FAST detector trained using the scheme of Rosten and Drummond[2], 
		    and the non-maximal suppression using FAST score

    -- SPHORB.h SPHORB.cpp
                    the SPHORB algorithm

    -- example1.cpp example2.cpp
                    two test cases



[1] J. Xiao, K. Ehinger, A. Oliva, and A. Torralba. Recognizing scene viewpoint 
    using panoramic place representation. In Proceedings of the IEEE Conference 
    on Computer Vision and Pattern Recognition (CVPR), pages 2695¨C2702, 2012.

[2] E. Rosten and T. Drummond. Machine learning for highspeed corner detection. 
    In Proceedings of the European Conference on Computer Vision (ECCV), 2006.



-------------------
Contact information
-------------------

For any questions, comments, bug reports or suggestions, 
please send email to Qiang Zhao at qiangzhao@tju.edu.cn.

