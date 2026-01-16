import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

def get_handmask(frame):
    hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 40], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    hsv_skinRegion = cv.inRange(hsv_img, lower, upper)
    return hsv_skinRegion
    
def trimming(img):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, k, iterations=4)
    return img

def get_convexAndcontours(img):
    contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=lambda x: cv.contourArea(x))
    hull_point = cv.convexHull(max_contour) #convexhull이란 boundingRect처럼 컨투어 좌표에서 움푹 들어간 곳을 펼쳐서 볼록하게 만들어서 그리는 것이고 contour에서 다 포함하는 최소한의 사각형을 그리는 것
    return hull_point, max_contour

def calculate_angle(start_point, end_point, farthest_point):
    a = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) #여기서 [0]은 x좌표 [1]은 y좌표를 뜻함
    b = np.sqrt((farthest_point[0] - start_point[0]) ** 2 + (farthest_point[1] - start_point[1]) ** 2)
    c = np.sqrt((end_point[0] - farthest_point[0]) ** 2 + (end_point[1] - farthest_point[1]) ** 2)
    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    return angle

def point_distance(a,b):
    distance = np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return distance

#여기서 두 element의 거리가 상당히 짧다면 한 녀석을 없애 줘야 하는데... start의 1index와 end의 0index의 비교를 해야하네?
#아냐 리스트안에 추가되는 순서가 start와 end순서가 다르니 즉, end추가 한 뒤에 start포인트가 들어올수도 있으니 전체 점을 다 구한후에 비교를 통해서 위치가 비슷한것을 제거하자 
def remove_closepoint(list1, list2):
    for point1 in list1:
        for index, point2 in enumerate(list2):
            if point_distance(point1,point2)<40:
                del list2[index]
    return list1+list2
    
color = (0,200,0)

while cap.isOpened():
    fingers_cnt = 0
    ret, frame = cap.read()
    if not ret:
        break
    mask = get_handmask(frame) #여기서 mask를 얻었으니까 기존의 이미지에서 원하는 색상을 얻고싶다면 and연산을 통해서 원하는 부분만 뽑아내면됨
    trimmed_mask = trimming(mask)
    
    try:
        hull_point, contour = get_convexAndcontours(trimmed_mask)
        cv.drawContours(frame, contour, -1, (255,255,0), 2)
        cv.drawContours(frame, [hull_point], -1, (0, 255, 255), 2)
        
        hull_index = cv.convexHull(contour, returnPoints=False) #마지막에 returnpoint을 적어주면 index값을 반환하는 군. contour의 좌표 즉 point가 아니라 index을 받는다는 것
        defects = cv.convexityDefects(contour, hull_index)
        # start_index, end_index, farthest_pt_index, fixpt_depth
        # 시작, 종료, 가장 먼 지점, 거리 수만은 contour의 좌표들 중에서 defects와 관련된 좌표들의 index을 알려주는 함수
        defect_cnt = 0
        fingers_start = []
        fingers_end = []

        for i in range(defects.shape[0]): #shape함수를 사용하면 결과값으로 크기, 정밀도, 채널을 나타내는군 // shape[0]으로 defects의 크기(갯수)를 찾아내었다
            start_index, end_index, farthest_pt_index, fixpt_depth = defects[i, 0] #그리고 이걸로 각각 defects의 순서중에서 하나하나 찾아가는 것인데 [[]] 이런식으로 list가 두겹이니까 요소중에 순서대로 하나 선택해서 [0]으로 리스트 하나를 벗겨버리는 것이다.
            # defects의 인덱스를 찾은것으로 각 지점의 좌표 구하기 
            start_point = tuple(contour[start_index][0])
            end_point = tuple(contour[end_index][0])
            farthest_point = tuple(contour[farthest_pt_index][0]) #여러 컨투어 의 좌표중에서 convex라인에서 제일로 먼 좌표의 index를 알게되었으니 그 녀석을 선택해주고 list를 하나 벗기기 위해 [0]을 해준다.
            dist = fixpt_depth/256.0 
            angle = calculate_angle(start_point, end_point, farthest_point)
            if dist>30 and angle<((np.pi)*24/40):
                fingers_start.append(start_point)
                #추가적으로 end와 defect의 거리가 충분히 멀다면 end도 손가락으로 쳐준다. 단 각도가 90도 보다 크다면 체크안한다.
                if point_distance(farthest_point, end_point)>100 and angle<np.pi/2:
                    fingers_end.append(end_point)
                cv.circle(frame, farthest_point, 3, (0,0,255), -1)
                defect_cnt+=1

        fingers = remove_closepoint(fingers_start, fingers_end)
        for i in fingers:
            cv.circle(frame, (i), 4, (0,255,0), 3)
        cv.putText(frame, 'defect: '+str(defect_cnt), (10, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
        cv.putText(frame, 'finger: '+str(len(fingers)), (10, 100), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
        
    except:
        pass    
    # cv.imshow('mask',hsv_skinRegion)
    cv.imshow('mask',mask)
    # cv.imshow('maskas',trimmed_mask)
    cv.imshow('frame',frame)
    if cv.waitKey(10) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
cv.waitKey(1)
cv.waitKey(1)