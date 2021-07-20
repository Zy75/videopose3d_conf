
What I want to do is change the viewport of a video as from behind the stage, so dancing learners can be freed from imagining 180 degrees rotation into the self-view.

How: I used the pose estimation AI. I used VideoPose3D from Facebook Research to get the 3D pose. Then I set the camera from behind the stage and render the 3D model.

Result: The pose could be extracted well in 3D way especially in slower motions, but it didn't work for detailed motions and quick rotations.

Changes I made: To set the camera behind the stage, I increased azimuth 180 and inverted elevation in VideoPose3D/common/visualization.py 
