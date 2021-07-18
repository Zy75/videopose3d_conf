
What I want to do is change the viewport of a video as from behind the stage, so dancing learners can be freed from imagining 180 degrees rotation into the self-view.

How: I used the pose estimation AI. I used VideoPose3D from Facebook Research. 

Result: The pose could be extracted well in 3D way especially in slower motions, but it didn't work for detailed motions and quick rotations.

Changes I made: I increased azimuth 180 and inverted elevation in VideoPose3D/common/visualization.py
