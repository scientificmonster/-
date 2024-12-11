import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class CoinDetector:
    def __init__(self, font_path="msyh.ttc"):
        self.font_path = font_path

    def add_text(self, img, text, position, text_color=(0, 0, 0), text_size=10):
        """在图像上绘制文字"""
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font_style = ImageFont.truetype(self.font_path, text_size, encoding="utf-8")
        draw.text(position, text, text_color, font=font_style)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def detect_coin_color(self, image, mask):
        """根据掩码区域的颜色判断硬币种类"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        coin_area = cv2.bitwise_and(hsv, hsv, mask=mask)
        non_zero = coin_area[mask > 0]
        if len(non_zero) > 0:
            mean_hsv = np.mean(non_zero, axis=0)
            hue, saturation = mean_hsv[0], mean_hsv[1]
            if 15 <= hue <= 45 and saturation >= 50:
                return "5"
        return "1"

    def detect_coins(self, image_path, scale_percent=50):
        """主函数：检测硬币并统计总金额"""
        # 读取原始图像
        original_image = cv2.imread(image_path)

        # 显示原始图像
        cv2.imshow("Original Image", original_image)
        cv2.waitKey(1000)

        image = self.resize_image(original_image, scale_percent)

        # 显示调整大小后的图像
        cv2.imshow("Resized Image", image)
        cv2.waitKey(1000)

        # 灰度处理
        gray = self.preprocess_image(image)

        # 显示灰度图
        cv2.imshow("Grayscale Image", gray)
        cv2.waitKey(1000)

        # 高斯模糊后的图像
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        cv2.imshow("Blurred Image", blurred)
        cv2.waitKey(1000)

        # 检测圆形
        circles = self.find_circles(blurred)

        # 创建边缘检测图像用于可视化
        edges = cv2.Canny(blurred, 50, 150)
        cv2.imshow("Edge Detection", edges)
        cv2.waitKey(1000)

        return self.process_circles(image, gray, circles)

    def resize_image(self, image, scale_percent):
        """调整图像大小以加速处理"""
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height))

    def preprocess_image(self, image):
        """图像预处理，转换为灰度图并进行模糊"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def find_circles(self, gray):
        """使用霍夫圆检测找到硬币"""
        return cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=80,
            param2=25,
            minRadius=15,
            maxRadius=80,
        )

    def process_circles(self, image, gray, circles):
        """处理检测到的圆并统计硬币类型和总金额"""
        total_amount = 0
        coin_counts = {"1": 0, "0.5": 0, "0.1": 0}
        masks = np.zeros_like(gray)

        # 复制原始图像用于绘制结果
        result_image = image.copy()
        mask_visualization = np.zeros_like(image)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circles = sorted(circles, key=lambda x: x[2], reverse=True)

            for x, y, r in circles:
                # 创建单个硬币的掩码
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                masks = cv2.add(masks, mask)

                # 可视化单个硬币掩码
                mask_color = np.zeros_like(image)
                cv2.circle(mask_color, (x, y), r, (0, 255, 0), -1)
                mask_visualization = cv2.addWeighted(mask_visualization, 0.5, mask_color, 0.5, 0)

                coin_color = self.detect_coin_color(image, mask)

                if r >= 35:
                    coin_type, amount, color = "1", 1.0, (0, 0, 0)  # 黑色
                elif coin_color == "5":
                    coin_type, amount, color = "0.5", 0.5, (0, 255, 255)  # 黄色
                else:
                    coin_type, amount, color = "0.1", 0.1, (0, 0, 255)  # 红色

                coin_counts[coin_type] += 1
                total_amount += amount
                cv2.circle(result_image, (x, y), r, color, 2)
                result_image = self.add_text(result_image, coin_type, (x - 25, y - r - 20), color, 20)

        # 显示所有中间处理结果
        cv2.imshow("Original Mask", masks)
        cv2.imshow("Mask Visualization", mask_visualization)
        cv2.imshow("Circle Detection Result", result_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        return coin_counts, total_amount


def print_coin_statistics(counts, total):
    """格式化并打印硬币统计结果"""
    print("\n硬币统计结果：")
    print("-" * 30)
    print(f"{'硬币类型':<10}{'数量':<10}")
    print("-" * 30)
    for coin_type, count in counts.items():
        print(f"{coin_type:<10}{count:<3}枚")
    print("-" * 30)
    print(f"{'总金额':<10}{total:.1f}元")
    print("-" * 30)


if __name__ == "__main__":
    image_path = r"E:\AllProject\pythonProject\cspic\image.png"
    detector = CoinDetector()
    counts, total = detector.detect_coins(image_path)

    print_coin_statistics(counts, total)