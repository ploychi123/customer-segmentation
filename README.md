# 📊 Customer Segmentation using K-Means and RFM Analysis

## 📌 Overview
โปรเจกต์นี้เป็นการทำ Customer Segmentation โดยใช้ RFM Analysis (Recency, Frequency, Monetary) ร่วมกับ K-Means Clustering เพื่อแบ่งกลุ่มลูกค้าตามพฤติกรรมการซื้อ และสร้าง Insight ที่สามารถนำไปใช้ในเชิงธุรกิจ เช่น การทำ Targeted Marketing และ Customer Retention Strategy

## 📁 Dataset
ใช้ชุดข้อมูลจาก https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset 
โดยข้อมูลมีทั้งหมด 18 คอลัมน์ 3,900 แถว โดยมีรายละเอียดคอลัมน์ ดังนี้
- Customer ID - Unique identifier for each customer
- Age - Age of the customer
- Gender - Gender of the customer (Male/Female)
- Item Purchased - The item purchased by the customer
- Category - Category of the item purchased
- Purchase Amount (USD) - The amount of the purchase in USD
- Location - Location where the purchase was made
- Size - Size of the purchased item
- Color - Color of the purchased item
- Season - Season during which the purchase was made
- Review Rating - Rating given by the customer for the purchased item
- Subscription Status - Indicates if the customer has a subscription (Yes/No)
- Shipping Type - Type of shipping chosen by the customer
- Discount Applied - Indicates if a discount was applied to the purchase (Yes/No)
- Promo Code Used - Indicates if a promo code was used for the purchase (Yes/No)
- Previous Purchases - The total count of transactions concluded by the customer at the store, excluding the ongoing transaction
- Payment Method - Customer's most preferred payment method
- Frequency of Purchases - Frequency at which the customer makes purchases (e.g., Weekly, Fortnightly, Monthly)

## ⚙️ Methodology
### 1. Data Cleaning & Preparation
- Import Data
- Missing Values check
- Duplicate Data check
พบว่าข้อมูลไม่มี Missing Values และ Duplicate data 

### 2.RFM Analysis
RFM Analysis คือเทคนิคการวิเคราะห์และแบ่งกลุ่มลูกค้าโดยใช้ข้อมูล 3 ปัจจัยหลัก ได้แก่

Recency (R): ระยะเวลาที่ลูกค้าซื้อสินค้าหรือใช้บริการครั้งล่าสุด ยิ่งใกล้เคียงปัจจุบันมาก ยิ่งแสดงว่าลูกค้ามีแนวโน้มจะกลับมาซื้อซ้ำสูง
Frequency (F): ความถี่ในการซื้อสินค้าหรือใช้บริการ ยิ่งลูกค้าซื้อบ่อย ยิ่งเป็นลูกค้าขาประจำ
Monetary (M): มูลค่ารวมของการซื้อสินค้าหรือใช้บริการ ยิ่งลูกค้ามียอดใช้จ่ายสูง ยิ่งเป็นกลุ่มลูกค้าที่สำคัญ

การวิเคราะห์นี้จะให้คะแนน (score) แต่ละลูกค้าในแต่ละปัจจัย เพื่อจัดกลุ่มลูกค้าออกเป็นกลุ่มต่าง ๆ ตามความสำคัญหรือคุณค่าที่ลูกค้าก่อให้เกิดกับธุรกิจ โดยช่วยให้ธุรกิจสามารถวางกลยุทธ์ทางการตลาดได้ตรงกลุ่มเป้าหมาย เช่น การออกโปรโมชั่น หรือการสื่อสารเฉพาะกลุ่มลูกค้าที่สำคัญ มุ่งเน้นใช้ทรัพยากรอย่างมีประสิทธิภาพ และรักษาฐานลูกค้าที่มีมูลค่าสูงไว้ได้ดีขึ้น

```python
# Map "Frequency of Purchases" → Recency Proxy
freq_map = {
    "Bi-Weekly": 1,       # 2 ครั้ง/สัปดาห์ → ถี่ที่สุด
    "Weekly": 2,          # ทุกสัปดาห์
    "Fortnightly": 3,     # ทุก 2 สัปดาห์
    "Monthly": 4,         # ทุกเดือน
    "Quarterly": 5,       # ทุก 3 เดือน
    "Every 3 Months": 5,  # = Quarterly
    "Annually": 6         # ปีละครั้ง → ห่างที่สุด
}
df["Recency Proxy"] = df["Frequency of Purchases"].map(freq_map)
```
เนื่องจาก
