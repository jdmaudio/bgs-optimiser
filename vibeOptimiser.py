import nlopt
import pysky360
import numpy as np
import cv2
import random
import metrics
import movingobjects

def load_video(video_path, num_frames=100, frameSize=(1024,1024)):
    
    cap = cv2.VideoCapture(video_path)

    # Reads the first N frames
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frameSize)
            frames.append(frame)
        else:
            break

    cap.release()
    frames = np.array(frames)
    return frames

def draw_synthetic_data(groundTruth, frame, circles):
    for circle in circles:
        circle.move()
        circle.draw(groundTruth, (255, 255, 255))
        circle.draw(frame, (10, 10, 20))
    groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)
    return frame, groundTruth


def objective_function(params, grad, args):

    frames = args
    threshold, BGSamples, requiredBGSample, learningRate = params[0], params[1], params[2], params[3]

    if int(learningRate) > int(BGSamples):
        learningRate = int(BGSamples)
    
    # Set BGS parameters
    bgsalgorithm = pysky360.Vibe()
    parameters = bgsalgorithm.getParameters()
    parameters.setThreshold(int(threshold))
    parameters.setBGSamples(int(BGSamples))
    parameters.setRequiredBGSamples(int(requiredBGSample))
    parameters.setLearningRate(int(learningRate))

    # Initialize variables
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    # Mask out periphery
    first_frame = frames[0]
    mask = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.uint8)
    radiusOfMask = int(min(first_frame.shape[1], first_frame.shape[0])*0.43)
    centerX = int(first_frame.shape[1] / 2)
    centerY = int(first_frame.shape[0] / 2)
    cv2.circle(mask, (centerX, centerY), radiusOfMask, (255,255,255), -1)

    # Moving objects    
    rng = random.seed(12345)
    circles = [movingobjects.MovingCircle(centerX, centerY, radiusOfMask) for i in range(16)]

    for frame in frames: 

        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
        groundTruth = np.zeros_like(frame_masked)
        frame, ground_truth = draw_synthetic_data(groundTruth, frame_masked, circles)
        foreground_mask = bgsalgorithm.apply(frame)

        cv2.imshow('frame', frame)
        cv2.imshow('foreground_mask', foreground_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cm = metrics.ConfusionMatrix(ground_truth, foreground_mask, mask)
        TP, TN, FP, FN = cm.get()

        # Update totals
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN
    
    temp = metrics.Metrics(total_TP, total_TN, total_FP, total_FN)
    print("Fitness (MCC): " + str(np.round(temp.MCC,4)))
    return temp.MCC

def my_constraint(x, grad):
    # Constraint 1: Param 3 <= Param 2
    if x[2] > x[1]:
        return -1.0
    # Constraint 2: Param 4 <= Param 2
    if x[3] >= x[1]:
        return -1.0
    # Constraint 3: 4 is power of two
    if not is_power_of_two(int(x[3])):
        return -1.0
    
    return 0.0

def is_power_of_two(n: int) -> bool:
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


video_file = "testBG.mp4"

# Read N frames from a video 
frames = load_video(video_file, 250, (800, 800))

# Run Global NLOPT optimiser
print("\nRunning global optimiser...")
opt = nlopt.opt(nlopt.GN_CRS2_LM, 4)
opt.set_max_objective(lambda x, grad: objective_function(x, grad, frames))
lower_bounds = [5, 2, 1, 2]
upper_bounds = [80, 32, 2, 16]
opt.set_lower_bounds(lower_bounds)
opt.set_upper_bounds(upper_bounds)
opt.set_maxeval(6)
initial_params = [random.randint(5, 80), random.randint(2, 32), random.randint(1, 2), random.randint(2, 16)]
result = opt.optimize(initial_params)
maxf = opt.last_optimum_value()
print("Best parameters from global search:", np.round(result))
print("Best score:", maxf)

# Run local NLOPT optimiser onb best
print("\nRunning local optimiser...")
opt = nlopt.opt(nlopt.LN_COBYLA, 4)
opt.set_max_objective(lambda x, grad: objective_function(x, grad, frames))
opt.set_lower_bounds(lower_bounds)
opt.set_upper_bounds(upper_bounds)
opt.set_maxeval(6)
opt.add_inequality_constraint(my_constraint)
result = opt.optimize(result)
maxf = opt.last_optimum_value()
print("Best parameters after local optimiser: ", np.round(result))
print("Best score: ", maxf)