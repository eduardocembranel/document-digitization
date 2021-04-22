import cv2
import numpy as np

def load_image(img_path):
    return cv2.imread(img_path, cv2.IMREAD_COLOR)

def write_image(img_path, img):
    cv2.imwrite(img_path, img)

def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def dilate(img, k=5):
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)
    
def erode(img, k=5):
    kernel = np.ones((k,k),np.uint8)
    return cv2.erode(img, kernel, iterations = 1)

def opening(img):
    kernel = np.ones((3, 3),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#n precisa disso se pa
def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def sharpen(img):
    sharpen_kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, sharpen_kernel)

def noise_removal(img, k=5):
    return cv2.medianBlur(img, k)

def clahe(img, clip=2.0, tile_grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(img)

def add_3_channels(img):
    img2 = np.zeros((img.shape) + (3,))
    img2[:,:,0] = img
    img2[:,:,1] = img
    img2[:,:,2] = img
    return img2

def deskew(img):
    gray = gray_scale(img)
    thresh = cv2.threshold(gray, 0, 255,
	    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def align_images(img1, img2, max_features=20000, good_match_percent=0.01):
    img1_gray = gray_scale(img1)
    img2_gray = gray_scale(img2)

    orb = cv2.ORB_create(max_features)
    key_points1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    key_points2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    matches.sort(key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    img_matches = cv2.drawMatches(img1, key_points1, img2, key_points2, matches, None)
    write_image('../Images/matches.png', img_matches)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = key_points1[match.queryIdx].pt
        points2[i, :] = key_points2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = img2.shape
    img1Reg = cv2.warpPerspective(img1, h, (width, height))

    return img1Reg, h