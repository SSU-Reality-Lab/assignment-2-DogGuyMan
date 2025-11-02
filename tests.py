import os, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import features

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def save_heatmap(array, title, filename, cmap='jet'):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_keypoints(image, keypoints, filename):
    vis = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), 2, (0,255,0), -1)
    os.makedirs('results', exist_ok=True)
    cv2.imwrite(filename, vis)

# -------------------------------------------------------------------
# 0️⃣ Load Images
# -------------------------------------------------------------------
img1 = cv2.imread('resources/yosemite1.jpg')
img2 = cv2.imread('resources/yosemite2.jpg')

gray1 = cv2.cvtColor(img1.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------
# 1️⃣ Feature Computation (TODO1~6)
# -------------------------------------------------------------------
HKD = features.HarrisKeypointDetector()
SFD = features.SimpleFeatureDescriptor()
MFD = features.MOPSFeatureDescriptor()

# TODO1
a1, b1 = HKD.computeHarrisValues(gray1)
a2, b2 = HKD.computeHarrisValues(gray2)

# TODO3
d1 = HKD.detectKeypoints(img1)
d2 = HKD.detectKeypoints(img2)

# Filter weak keypoints
d1 = [kp for kp in d1 if kp.response > 0.01]
d2 = [kp for kp in d2 if kp.response > 0.01]

# TODO4~6
desc_simple_1 = SFD.describeFeatures(img1, d1)
desc_simple_2 = SFD.describeFeatures(img2, d2)
desc_mops_1 = MFD.describeFeatures(img1, d1)
desc_mops_2 = MFD.describeFeatures(img2, d2)

# -------------------------------------------------------------------
# 2️⃣ Visualization (TODO1, TODO3)
# -------------------------------------------------------------------
save_heatmap(a1, "Image1 - TODO1 Harris Strength", "results/img1_TODO1_harris_strength.png")
save_heatmap(a2, "Image2 - TODO1 Harris Strength", "results/img2_TODO1_harris_strength.png")

save_keypoints(img1, d1, "results/img1_TODO3_detected_keypoints.png")
save_keypoints(img2, d2, "results/img2_TODO3_detected_keypoints.png")

print("✅ Saved TODO1 & TODO3 visualizations.")

# -------------------------------------------------------------------
# 3️⃣ Matching (TODO7 - SSD, TODO8 - Ratio)
# -------------------------------------------------------------------
matcher_ssd = features.SSDFeatureMatcher()
matcher_ratio = features.RatioFeatureMatcher()

# ------------------------------
# TODO7 - SSD matching
# ------------------------------
# Step 1. SSD matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.

# "__________"를 채워야 한다.

matches_ssd = matcher_ssd.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. 거리(distance)를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ssd = sorted(matches_ssd, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ssd_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ssd, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO7_SSD_matches.png", ssd_vis)
print("✅ TODO7 (SSD) match result saved → results/TODO7_SSD_matches.png")

# ------------------------------
# TODO8 - Ratio matching
# ------------------------------
# Step 1. Ratio matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ratio = matcher_ratio.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. distance를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ratio = sorted(matches_ratio, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ratio_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ratio, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO8_Ratio_matches.png", ratio_vis)
print("✅ TODO8 (Ratio) match result saved → results/TODO8_Ratio_matches.png")

print("🎯 All TODO1–8 visualizations done! Files saved in 'results/'")

'''
8단계의 매칭이 더 잘된 이유

SSD는 
비슷한 디스크립터가 여럿 존재하거나, 아예 대응점이 없는경우에도
오직 맹목적인 거리 우선을 매칭을 진행하기 때문에
매칭이 발생하기 떄문에 Outlier가 생긴다. 
그리고 이것은 추후 이미지 정렬에 있어서 
오차 제곱합의 결과는 Outlier에 매우 민감하므로
잘못된 결과를 낼 가능성이 있다

하지만 Ratio Test는 후보1, 2(일명 최근접 후보들) 를 나눠 
거리 비(Ratio)를 통해 서로의 거리 유사도 검사를 진행 후, 
만약 1.0f 에 가깝다면 매우 모호 한 것이고, 
특정 threshold 이하라면 거리상으로 모호하지 않고 확실히 구별될 수 있다는 의미고,
이러한 케이스만 남겨, 필터링을 진행하기 때문에 Outlier를 상당수 제거할 수 있다.
'''