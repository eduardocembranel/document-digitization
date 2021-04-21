import cv2
import numpy as np

def load_image(img_path):
    return cv2.imread(img_path, cv2.IMREAD_COLOR)

def write_image(img_path, img):
    cv2.imwrite(img_path, img)

def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def deskewing(img1, img2, max_features=500, good_match_percent=0.15):
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