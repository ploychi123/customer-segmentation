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
### 1. Data Cleaning
- Import Data
- Missing Values check
- Duplicate Data check

พบว่าข้อมูลไม่มี Missing Values และ Duplicate data 

### 2. RFM Analysis
RFM Analysis คือเทคนิคการวิเคราะห์และแบ่งกลุ่มลูกค้าโดยใช้ข้อมูล 3 ปัจจัยหลัก ได้แก่

Recency (R): ระยะเวลาที่ลูกค้าซื้อสินค้าหรือใช้บริการครั้งล่าสุด ยิ่งใกล้เคียงปัจจุบันมาก ยิ่งแสดงว่าลูกค้ามีแนวโน้มจะกลับมาซื้อซ้ำสูง
Frequency (F): ความถี่ในการซื้อสินค้าหรือใช้บริการ ยิ่งลูกค้าซื้อบ่อย ยิ่งเป็นลูกค้าขาประจำ
Monetary (M): มูลค่ารวมของการซื้อสินค้าหรือใช้บริการ ยิ่งลูกค้ามียอดใช้จ่ายสูง ยิ่งเป็นกลุ่มลูกค้าที่สำคัญ

การวิเคราะห์นี้จะให้คะแนน (score) แต่ละลูกค้าในแต่ละปัจจัย เพื่อจัดกลุ่มลูกค้าออกเป็นกลุ่มต่าง ๆ ตามความสำคัญหรือคุณค่าที่ลูกค้าก่อให้เกิดกับธุรกิจ โดยช่วยให้ธุรกิจสามารถวางกลยุทธ์ทางการตลาดได้ตรงกลุ่มเป้าหมาย เช่น การออกโปรโมชั่น หรือการสื่อสารเฉพาะกลุ่มลูกค้าที่สำคัญ มุ่งเน้นใช้ทรัพยากรอย่างมีประสิทธิภาพ และรักษาฐานลูกค้าที่มีมูลค่าสูงไว้ได้ดีขึ้น

dataset นี้ ไม่มีข้อมูลวันที่ซื้อ (Purchase Date/InvoiceDate) → ทำ RFM มาตรฐานแท้ ๆ ไม่ได้
แต่สามารถทำ Pseudo-RFM โดยใช้ตัวแปรที่ dataset มีดังนี้:

Monetary (M) → Purchase Amount (USD) → รวมมูลค่าซื้อ

Frequency (F) → Previous Purchases → จำนวนครั้งที่เคยซื้อ

Recency (R - Proxy) → Frequency of Purchases (Weekly, Monthly …) → map เป็นตัวเลข

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
จะได้ตัวเลขที่แทนความถี่ในการซื้อ ยิ่งค่าน้อย ยิ่งมีความถี่มาก

```python
# Feature Engineering
# RFM features (Recency, Frequency, Monetary)
rfm_pseudo = df.groupby("Customer ID").agg({
    "Purchase Amount (USD)": "sum", #Monetary ลูกค้าใช้เงินเท่าไร
    "Previous Purchases": "max" , #Frequency ลูกค้าซื้อกี่ครั้ง ภายในช่วงเวลาที่กำหนด
    "Recency Proxy": "min" #Recency Proxy ลูกค้าซื้อครั้งล่าสุดห่างจากปัจจุบันกี่วัน/สัปดาห์/เดือน
}).reset_index()

rfm_pseudo.rename(columns={
    "Purchase Amount (USD)": "Monetary",
    "Previous Purchases": "Frequency",
    "Recency Proxy": "Recency"
}, inplace=True)
```
<img width="525" height="458" alt="image" src="https://github.com/user-attachments/assets/da8ff590-b433-42ee-b753-a274bab5c4f0" />

เนื่องจาก Customer ID ในชุดข้อมูลนี้ไม่ซ้ำเลยจึงทำให้ได้ตาราง RFM Pseudo (Monetary, Frequency, Recency) ต่อคน เป็น 3,900 คน

### 3. Exploratory Data Analysis (EDA)
#### 3.1 Distribution of R, F, M

```python 
# วิเคราะห์การกระจายของ R, F, M
plt.figure(figsize=(15,4))
for i, col in enumerate(['Recency','Frequency','Monetary']):
    plt.subplot(1, 3, i+1)
    sns.histplot(rfm_pseudo[col], bins=30, kde=True, color="skyblue")
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()
```

<img width="1490" height="390" alt="image" src="https://github.com/user-attachments/assets/0dc316bf-6985-44ff-921d-2f3078f34b8e" />

1. Distribution of Recency
Recency = จำนวนวันที่ผ่านไปนับตั้งแต่ลูกค้าซื้อครั้งล่าสุด
กราฟแสดงว่ามีการกระจายตัวไม่เท่ากัน → โดยเฉพาะที่ค่า 5 มีจำนวนลูกค้ามากกว่าช่วงอื่น ๆ อย่างชัดเจน
หมายความว่า ลูกค้าส่วนใหญ่เพิ่งซื้อสินค้าเมื่อ 5 วันก่อน → เป็นลูกค้าที่ค่อนข้าง active แสดงว่าช่วงเวลานี้เป็นช่วงที่ลูกค้า engage เยอะ อาจสัมพันธ์กับโปรโมชั่นหรือพฤติกรรมการซื้อซ้ำ

2. Distribution of Frequency
Frequency = ความถี่ในการซื้อของลูกค้า
กราฟกระจายค่อนข้างสม่ำเสมอ (uniform-like) → ลูกค้าซื้อสินค้าในหลายระดับความถี่ตั้งแต่ 1–50 ครั้ง
ไม่มี peak ที่เด่นชัด → หมายถึงธุรกิจมีลูกค้าที่ พฤติกรรมการซื้อหลากหลาย (บางคนซื้อน้อยครั้ง บางคนซื้อบ่อย)สามารถแบ่งกลุ่มลูกค้าได้ เช่น “ซื้อไม่บ่อย” vs “ซื้อซ้ำบ่อย” เพื่อนำไปทำ Loyalty program

3. Distribution of Monetary
Monetary = จำนวนเงินที่ลูกค้าใช้จ่ายทั้งหมด
กราฟแสดงว่าลูกค้ามีการใช้จ่ายตั้งแต่ 20–100 USD ค่อนข้างกระจายสม่ำเสมอเช่นเดียวกับ Frequency
ไม่มีการกระจุกที่ระดับใดระดับหนึ่ง → หมายถึงลูกค้า ใช้จ่ายหลายระดับ ทั้งลูกค้าขนาดเล็ก (20–40) และลูกค้าขนาดใหญ่ (80–100) กลุ่มใช้จ่ายสูงควรได้รับการดูแลพิเศษ (High Value Customers) ในขณะที่กลุ่มใช้จ่ายต่ำอาจต้องใช้ โปรโมชั่น/ส่วนลด กระตุ้น

สรุป 
Recency: ลูกค้าส่วนใหญ่เพิ่งซื้อเมื่อเร็ว ๆ นี้ (Active Customers)
Frequency: ลูกค้ามีพฤติกรรมการซื้อที่หลากหลาย ตั้งแต่ซื้อน้อยครั้งจนถึงบ่อยครั้ง
Monetary: ลูกค้ามีการใช้จ่ายหลากหลาย ทำให้สามารถแบ่งได้เป็นกลุ่ม High Value และ Low Value

#### 3.2 check outlier with boxplot

```python
cols = ['Age','Purchase Amount (USD)','Previous Purchases','Review Rating']

for col in cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
```
<img width="532" height="353" alt="image" src="https://github.com/user-attachments/assets/32b67de3-80f2-40f8-84a9-cc05eb9c41a9" />

จาก boxplot Age ข้อมูลอายุมีการกระจายตัวค่อนข้างสมดุล ไม่พบค่าผิดปกติ กลุ่มลูกค้าส่วนใหญ่เป็น วัยทำงานถึงวัยกลางคน (30–57 ปี)

<img width="540" height="353" alt="image" src="https://github.com/user-attachments/assets/43a5b3e2-985d-4825-b278-fee541043a91" />

จาก boxplot Purchase Amount (USD) ไม่พบค่าผิดปกติ ลูกค้าส่วนใหญ่ซื้อสินค้าในช่วง 40–80 USD

<img width="532" height="353" alt="image" src="https://github.com/user-attachments/assets/47744f12-7d87-4b40-bee3-b25c0f950c52" />

จาก boxplot Previous Purchases ไม่พบค่าผิดปกติ ลูกค้าส่วนใหญ่เคยซื้อสินค้าอยู่ในช่วง 12–38 ครั้ง กลุ่มที่มีการซื้อซ้ำสูง (ใกล้ 50 ครั้ง) ถือเป็นลูกค้าประจำ (Loyal Customers) ที่ควรได้รับสิทธิพิเศษหรือโปรโมชั่นเฉพาะกลุ่ม ลูกค้าที่อยู่ในช่วงต่ำ (1–12 ครั้ง) อาจเป็นกลุ่ม ใหม่ (New Customers) ที่ต้องมีแคมเปญเพื่อกระตุ้นให้ซื้อซ้ำ

<img width="536" height="353" alt="image" src="https://github.com/user-attachments/assets/8d389afb-4204-4509-b7f5-61d86e61c1f6" />

จาก boxplot Review Rating ไม่พบค่าผิดปกติ คะแนนรีวิวของลูกค้าส่วนใหญ่กระจุกอยู่ระหว่าง 3.1 – 4.4 คะแนน → ถือว่าลูกค้าพอใจในระดับปานกลางถึงดี มีลูกค้าส่วนน้อยที่รีวิวต่ำกว่า 3.0 → อาจสะท้อนปัญหาด้านคุณภาพหรือบริการบางส่วน ลูกค้าที่ให้คะแนนสูงใกล้ 5.0 บ่งบอกถึงกลุ่ม High Satisfaction ที่สามารถใช้เป็น Promoter ในเชิงการตลาด (Word of Mouth / Loyalty Program)

### 4. Data Preparation for custering
#### 4.1 Scale Data
การ Scale Data ก่อนเข้า k-means มีเหตุผลเพื่อให้องค์ประกอบข้อมูลแต่ละคุณลักษณะอยู่ในมาตรฐานเดียวกัน เนื่องจาก k-means ใช้การคำนวณระยะห่างแบบ Euclidean distance ในการจับกลุ่มข้อมูล ถ้าหากข้อมูลแต่ละมิติมีช่วงค่าที่แตกต่างกันมาก จะทำให้ข้อมูลบางมิติที่มีค่าใหญ่กว่าครอบงำการคำนวณระยะทาง ทำให้ผลการแบ่งกลุ่มไม่ถูกต้องหรือผิดเพี้ยนไป การทำ Scale หรือ normalization ของข้อมูล เช่น การปรับค่าให้มีค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1 (standardization) จะช่วยให้ข้อมูลทั้งหมดมีน้ำหนักเท่าเทียมกันและถูกต้องยิ่งขึ้นในขั้นตอน clustering

```python
# Encode Categorical Variables (สำหรับ Clustering) เนื่องจาก K-Means ไม่สามารถทำงานกับข้อมูลที่เป็นตัวอักษรได้
cat_cols = ["Gender", "Category", "Payment Method", "Season", "Shipping Type","Discount Applied", "Promo Code Used"]

# dictionary สำหรับเก็บ mapping
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# แสดง mapping ของทุกคอลัมน์
for col, mapping in encoders.items():
    print(f"{col}: {mapping}")

# เลือกเฉพาะคอลัมน์ที่ใช้ทำ Clustering
features = ["Age","Discount Applied", "Promo Code Used","Gender",
            "Category", "Payment Method", "Season", "Shipping Type"]

# รวม RFM Table (พฤติกรรมการซื้อ) + Features เสริม (จาก df) กลายเป็น Feature Matrix (X) ที่มีแต่ตัวเลข พร้อมใช้เป็น Input ของ K-Means
X = rfm_pseudo.merge(df.groupby("Customer ID")[features].mean().reset_index(), on="Customer ID", how="left").drop(columns=["Customer ID"])

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
#### 4.2 Principal Component Analysis (PCA)
Principal Component Analysis (PCA) คือเทคนิคทางสถิติที่ใช้สำหรับลดจำนวนมิติของข้อมูล (Dimensionality Reduction) โดยเปลี่ยนแปลงข้อมูลที่มีจำนวนตัวแปรจำนวนมากให้เป็นชุดของตัวแปรใหม่ที่เรียกว่า "องค์ประกอบหลัก" (Principal Components) ซึ่งเป็นตัวแปรที่ไม่มีความสัมพันธ์กัน (Uncorrelated) และสามารถแสดงความแปรปรวนของข้อมูลได้มากที่สุดในแต่ละองค์ประกอบ เทคนิคนี้ช่วยลดความซับซ้อนของข้อมูล โดยยังคงรักษาแง่มุมสำคัญของข้อมูลไว้ และมักถูกใช้เพื่อลดจำนวนฟีเจอร์ก่อนนำข้อมูลไปสร้างโมเดล Machine Learning หรือนำไปใช้แสดงภาพข้อมูลในมิติต่ำลง

```python
# ทำ PCA 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ใช้ X.columns แทน features
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1','PC2'],
    index=X.columns
)

print(loadings)
```
<img width="377" height="337" alt="image" src="https://github.com/user-attachments/assets/50a73f48-e478-4873-9ac8-d3c6b374347c" />

ค่าที่แสดงในตารางนี้เป็น Loading Values ซึ่งบอกว่า แต่ละตัวแปรมีน้ำหนัก (contribution) ต่อการสร้างคอมโพเนนต์แต่ละตัวเท่าใด

PC1
ตัวแปรที่มีค่า loading สูงใน PC1 ได้แก่:
- Discount Applied (0.613594)
- Promo Code Used (0.613594)
- Gender (0.494381)
แสดงว่า PC1 ส่วนใหญ่ถูกอธิบายโดย พฤติกรรมด้านโปรโมชั่นและคุณลักษณะทางเพศ
ดังนั้น PC1 สามารถตีความได้ว่าเป็น แกนที่สะท้อนพฤติกรรมการตอบสนองต่อโปรโมชั่น (Promotion-driven Behavior)

PC2
ตัวแปรที่มีค่า loading สูงใน PC2 ได้แก่:
- Cluster (0.700666)
- Age (-0.688040)
- Season (0.115778)
- Shipping Type (0.092142)
แสดงว่า PC2 ส่วนใหญ่ถูกอธิบายโดย อายุของลูกค้า และปัจจัยด้านฤดูกาลและการขนส่ง
โดยเฉพาะค่า Age (-0.688040) ที่เป็นค่าลบสูง แสดงว่าลูกค้าที่อายุน้อยกว่าจะสัมพันธ์กับค่า PC2 สูง และในทางกลับกันลูกค้าที่อายุมากกว่าจะสัมพันธ์กับค่า PC2 ต่ำ
ดังนั้น PC2 สามารถตีความได้ว่าเป็น แกนที่สะท้อนคุณลักษณะด้านประชากรศาสตร์และฤดูกาล (Demographic & Seasonal Behavior)

#### 4.3 Principal Component Analysis (PCA)
Elbow Method (วิธีข้อศอก) เป็นเทคนิคที่ใช้ในการกำหนดจำนวนกลุ่ม (k) ที่เหมาะสมในกระบวนการแบ่งกลุ่มด้วย K-means โดยวิธีนี้จะพล็อตกราฟระหว่างจำนวนกลุ่ม (k) กับค่า Within-Cluster Sum of Squares (WCSS) หรือผลรวมของระยะห่างกำลังสองระหว่างจุดข้อมูลภายในแต่ละกลุ่มกับจุดศูนย์กลางของกลุ่มนั้น ๆ ซึ่ง WCSS จะลดลงเรื่อย ๆ เมื่อจำนวนกลุ่มเพิ่มขึ้น เพราะข้อมูลถูกแบ่งกลุ่มละเอียดขึ้น

หลักการของ Elbow Method คือดูจุดที่กราฟเริ่มโค้งงอคล้าย "ข้อศอก" (elbow) จุดนี้คือจำนวนกลุ่มที่เหมาะสมที่สุด เพราะหลังจากจุดนี้การเพิ่มจำนวนกลุ่มมากขึ้นจะไม่ทำให้ WCSS ลดลงอย่างมีนัยสำคัญอีกต่อไป ซึ่งหมายความว่าการแบ่งกลุ่มที่มากกว่านี้จะไม่ได้ช่วยปรับปรุงผลลัพธ์อย่างมีประสิทธิภาพมากนัก

สรุปคือ Elbow Method ใช้เพื่อตัดสินใจเลือกค่า k ที่เหมาะสมสำหรับ K-means clustering โดยพิจารณาจากจุดที่ลดค่า WCSS อย่างมีนัยสำคัญก่อนที่จะเข้าสู่ช่วงที่ลดลงช้า ๆ

```python
wcss = []
K = range(1, 10)  

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X)
    wcss.append(km.inertia_)  # inertia = WCSS

plt.figure(figsize=(6,4))
plt.plot(K, wcss, marker='o')
plt.title('Elbow Method (WCSS vs k)')
plt.xlabel('k')
plt.ylabel('WCSS / Inertia')
plt.xticks(K)
plt.show()
```

<img width="536" height="391" alt="image" src="https://github.com/user-attachments/assets/6f7889b9-dca2-454c-97d8-150e1c9e48fc" />

จากกราฟ Elbow Method (WCSS vs k) หลังจาก k=3 ความชันเริ่มค่อย ๆ ลดลง (เส้นโค้งแบนลง) จำนวนคลัสเตอร์ที่เหมาะสม ≈ 3

### 5. Model

```python
#K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
rfm_pseudo['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualization of Clusters
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=rfm_pseudo['Cluster'], cmap="tab10")
plt.title("Customer Segmentation (K-Means)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
```
<img width="689" height="545" alt="image" src="https://github.com/user-attachments/assets/e85da0be-2b33-4169-b969-6afa4a590cb2" />

PC1 แทนพฤติกรรมการใช้ โปรโมชั่น/ส่วนลด และอาจมีความแตกต่างด้าน เพศ
PC2 แทน โครงสร้างกลุ่มลูกค้า ที่เชื่อมโยงกับ อายุ และพฤติกรรมการซื้อในฤดูกาล/การจัดส่ง

Cluster 0 (ซ้ายสุด, สีน้ำตาล)
มีค่าน้อยใน PC1 แสดงว่าไม่ค่อยใช้โปรโมชั่น/โค้ดส่วนลด อาจเป็นลูกค้ากลุ่มที่ซื้อสินค้าโดยไม่เน้นราคา (Price-insensitive) หรือซื้อด้วยความจำเป็น ลูกค้าที่อาจไม่ถูกกระตุ้นด้วยโปรโมชั่น แต่ซื้อเพราะ Value ของสินค้าเอง

Cluster 1 (ขวาบน, สีฟ้า)
มีค่า PC1 สูง แสดงว่าใช้ลูกค้าโปรโมชั่น/ส่วนลดบ่อยค่า ,PC2 สูงและสัมพันธ์กับ Cluster และอายุ อาจเป็นกลุ่ม ลูกค้าอายุน้อย ที่ตอบสนองต่อโปรโมชัน Promotional Buyers → กลุ่มที่ควรกระตุ้นผ่านคูปอง/โค้ดลดราคา

Cluster 3 (ขวาล่าง, สีน้ำเงิน)
PC1 สูง  แสดงว่าใช้โปรโมชั่นเช่นเดียวกับ Cluster ขวาบนแต่ PC2 ต่ำ (สัมพันธ์กับอายุ -0.68) → น่าจะเป็นกลุ่ม ลูกค้าอายุมากกว่าซื้อผ่านส่วนลดเช่นกัน แต่พฤติกรรมอาจต่างจากกลุ่มบน (เช่น สนใจโปรโมชั่นในฤดูกาลเฉพาะ) Mature Promotional Buyers  ควรใช้โปรแบบ Seasonal / Shipping Type ที่เหมาะสม

สรุป
PC1: อธิบายพฤติกรรมการซื้อโดยใช้ โปรโมชั่นและโค้ดส่วนลด (Promotional Behavior)
PC2: แยกกลุ่มตาม อายุและโครงสร้างคลัสเตอร์ → ลูกค้าอายุน้อย vs อายุมาก

การรวมกัน:
Cluster 1 = Low Promo Usage (ไม่เน้นส่วนลด)
Cluster 2 = Young Promo Shoppers (ตอบสนองโปรมาก)
Cluster 3 = Older Promo Shoppers (ใช้โปรเช่นกันแต่พฤติกรรมต่างไป)








