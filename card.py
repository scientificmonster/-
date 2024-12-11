import cv2
import numpy as np

#该函数唯一作用是用于展示处理图片的全貌,实际应用时不需要使用
def show_image_with_resize(window_name, image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height, 1)  # 确保不会放大
    if scaling_factor < 1:
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    cv2.imshow(window_name, resized_image)
def area():
    """计算答题区域坐标"""
    box_width, box_height = 180, 960  # 单个答题区域的宽高
    y_offset = 570  # 初始的 Y 偏移
    coords = []
    for i in range(4):  # 假设有4列答题区域
        x_start = 140 + i * (box_width + 40)  # 每列的 X 偏移
        coords.append([
            [x_start, y_offset],
            [x_start + box_width, y_offset],
            [x_start, y_offset + box_height],
            [x_start + box_width, y_offset + box_height]
        ])
    return coords

class AnswerCardProcessor:
    """处理答题卡的检测与校正"""
    @staticmethod
    def detect_and_correct_card(image):
        """检测答题卡并校正方向"""
        # Step 1: 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        show_image_with_resize("Gray Image", gray)

        # Step 2: 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        show_image_with_resize("Blurred Image", blurred)

        # Step 3: 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        show_image_with_resize("Edges Detected", edges)

        # Step 4: 膨胀操作
        dilated = cv2.dilate(edges, None, iterations=2)
        show_image_with_resize("Dilated Edges", dilated)

        # Step 5: 找到最大轮廓并计算矩形
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)

        # Step 6: 调整角度确保竖直
        width, height = rect[1]
        angle = rect[2]
        if width > height:
            angle += 90 if angle < 0 else -90
        elif angle < -45:
            angle = 90 + angle

        # Step 7: 校正图像
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        show_image_with_resize("Step 7: Rotated Image", rotated)

        # Step 8: 计算旋转后的轮廓位置
        box = np.intp(cv2.boxPoints(rect))
        rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]

        # Step 9: 平移图像
        current_top_left = np.min(rotated_box, axis=0)
        dx, dy = 100 - current_top_left[0], 550 - current_top_left[1]
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        result = cv2.warpAffine(rotated, translation_matrix, (image.shape[1], image.shape[0]))
        show_image_with_resize("Step 9: Translated Image", result)

        return result



class AnswerAnalyzer:
    """分析答题卡上的答案"""
    def __init__(self, box_coords):
        self.box_coords = box_coords
    @staticmethod
    def preprocess_image(image):
        """将图像转换为二值化形式"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        return binary
    def analyze_answers(self, image):
        """分析答案区域并返回答题情况"""
        binary = self.preprocess_image(image)
        result = image.copy()
        answers = {}

        for box_idx, points in enumerate(self.box_coords):
            x1, y1 = points[0]
            x2, y2 = points[1]
            _, y3 = points[2]
            roi = binary[y1:y3, x1:x2]
            box_width = x2 - x1

            # 找轮廓
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            marks = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 20 < area < 350:
                    M = cv2.moments(cnt)
                    marks.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

            if marks:
                marks.sort(key=lambda x: x[1])
                question_marks = []
                current_y = marks[0][1]
                current_group = []

                # 分组
                for cx, cy in marks:
                    if abs(cy - current_y) > 20:
                        if current_group:
                            question_marks.append(current_group)
                        current_group = [(cx, cy)]
                        current_y = cy
                    else:
                        current_group.append((cx, cy))

                if current_group:
                    question_marks.append(current_group)

                # 提取每道题的答案
                for q_idx, q_marks in enumerate(question_marks):
                    question_num = q_idx + 1 + (30 * box_idx if box_idx < 3 else 90)
                    options = []
                    for cx, _ in q_marks:
                        option_idx = int((cx * 4) / box_width)
                        if 0 <= option_idx < 4:
                            options.append(chr(65 + option_idx))
                    if options:
                        answers[question_num] = sorted(options)

                    # 在图上标记
                    for cx, cy in q_marks:
                        center_x = x1 + cx
                        center_y = y1 + cy
                        cv2.circle(result, (center_x, center_y), 8, (0, 255, 0), -1)

        return result, answers


class AnswerCardSystem:
    """主处理系统"""
    def __init__(self, box_coords, image_paths):
        self.card_processor = AnswerCardProcessor()
        self.analyzer = AnswerAnalyzer(box_coords)
        self.image_paths = image_paths
    def process(self):
        for i, path in enumerate(self.image_paths):
            img = cv2.imread(path)
            img = cv2.resize(img, (1200, 1700), interpolation=cv2.INTER_LANCZOS4)

            # 检测和校正答题卡
            paper_img = self.card_processor.detect_and_correct_card(img)

            # 分析答案
            marked_img, answers = self.analyzer.analyze_answers(paper_img)

            # 打印结果
            print(f"\n{'=' * 30}\n图片 {i + 1} 的答题情况：\n{'=' * 30}")
            if not answers:
                print("未检测到答案。\n")
            else:
                line_width = 5  # 每行显示的题目数量
                sorted_questions = sorted(answers.keys())
                for idx, q_num in enumerate(sorted_questions):
                    print(f"第{q_num:3d}题: {', '.join(answers[q_num]):<10}", end="\t")
                    if (idx + 1) % line_width == 0:
                        print()  # 换行
                if len(sorted_questions) % line_width != 0:
                    print()  # 最后一行换行

            # 显示标记的图片
            display_img = cv2.resize(marked_img, (600, int(marked_img.shape[0] * 600 / marked_img.shape[1])))
            cv2.imshow(f"Image {i + 1}", display_img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    box_coords = area()
    # 图片路径
    image_paths = [
        r'E:\AllProject\pythonProject\cspic\ans1.jpg',
        r'E:\AllProject\pythonProject\cspic\ans2.jpg',
    ]
    system = AnswerCardSystem(box_coords, image_paths)
    system.process()
