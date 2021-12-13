# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)
            
def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)

def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    
    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']
    
    i = 0
    with sp.Popen(command, stdout = sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w*h*3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
            
                
                
    
def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index+2, projection='3d')
#Zy
        ax.view_init(elev=-15., azim= azim + 180)

        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius/2, radius/2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
        
        keypoints = keypoints[input_video_skip:] # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]
        
        if fps is None:
            fps = get_fps(input_video_path)
    
    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
   
    eyeplotA = None
    tailplotA = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points, eyeplotA,tailplotA

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'

        def tailplot(pos7,pos8,pos9,pos10):

            j7to8 = pos8 - pos7
            j8to9 = pos9 - pos8

            tmp1 = np.cross( j7to8, j8to9 )

            if np.linalg.norm( tmp1 ) <= 0.00001:
                print("Short Vector")

            vside1 = tmp1 / np.linalg.norm( tmp1 )

            tailc = pos8 - 2.0 * j8to9

            tails1 = tailc + vside1 * 0.03
            tails2 = tailc - vside1 * 0.03

            return [ ax.plot([pos8[0], tails1[0]],
                     [pos8[1], tails1[1]],
                     [pos8[2], tails1[2]], zdir='z', c='red'),
                     ax.plot([pos8[0], tails2[0]],
                     [pos8[1], tails2[1]],
                     [pos8[2], tails2[2]], zdir='z', c='red'),
                    ax.plot([tails2[0], tails1[0]],
                     [tails2[1], tails1[1]],
                     [tails2[2], tails1[2]], zdir='z', c='red') ]

        def eyeplot(pos7,pos8,pos9,pos10):

            j7to8 = pos8 - pos7
            j8to9 = pos9 - pos8

            tmp1 = np.cross( j7to8, j8to9 )

            if np.linalg.norm( tmp1 ) <= 0.00001:
                print("Short Vector")
            
            vside1 = tmp1 / np.linalg.norm( tmp1 )
            
            center910 = ( pos9 + pos10 ) / 2.0

            a1 = 15.0
            a2 = 10.0

            eye1in = center910 + vside1 / a1
            eye2in = center910 - vside1 / a1
            eye1out = center910 + vside1 / a2
            eye2out = center910 - vside1 / a2

            e1in = eye1in.tolist()
            e2in = eye2in.tolist()
            e1out = eye1out.tolist()
            e2out = eye2out.tolist()

            return [ ax.plot([e1in[0], e1out[0]],
                     [e1in[1], e1out[1]],
                     [e1in[2], e1out[2]], zdir='z', c='black'),
                    ax.plot([e2in[0], e2out[0]],
                     [e2in[1], e2out[1]],
                     [e2in[2], e2out[2]], zdir='z', c='black') ]


        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                    
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                
            pos7 =  [ pos[ 7,0], pos[ 7,1], pos[ 7,2] ]
            pos8 =  [ pos[ 8,0], pos[ 8,1], pos[ 8,2] ]
            pos9 =  [ pos[ 9,0], pos[ 9,1], pos[ 9,2] ]
            pos10 = [ pos[10,0], pos[10,1], pos[10,2] ]

            eyeplotA = eyeplot( np.array(pos7) , np.array(pos8) , np.array(pos9) , np.array(pos10) )           
            tailplotA = tailplot( np.array(pos7) , np.array(pos8) , np.array(pos9) , np.array(pos10) )           

            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                           [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')

            eyeplotA[0][0].remove()
            eyeplotA[1][0].remove()

            tailplotA[0][0].remove()
            tailplotA[1][0].remove()
            tailplotA[2][0].remove()

            pos7 =  [ pos[ 7,0], pos[ 7,1], pos[ 7,2] ]
            pos8 =  [ pos[ 8,0], pos[ 8,1], pos[ 8,2] ]
            pos9 =  [ pos[ 9,0], pos[ 9,1], pos[ 9,2] ]
            pos10 = [ pos[10,0], pos[10,1], pos[10,2] ]

            eyeplotA = eyeplot( np.array(pos7) , np.array(pos8) , np.array(pos9) , np.array(pos10) )           
            tailplotA = tailplot( np.array(pos7) , np.array(pos8) , np.array(pos9) , np.array(pos10) )           

            points.set_offsets(keypoints[i])
        
        print('{}/{}      '.format(i, limit), end='\r')
        

    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()
