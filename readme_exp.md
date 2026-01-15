# 🧒 Childcare Overview

Childcare refers to the care and supervision of children, typically from infancy through early school age, provided by parents, relatives, caregivers, or professional institutions. It can take the form of informal home-based care or formal institutional care (nurseries, preschools, kindergartens, or after-school programs).

---

## 📌 Types of Childcare

1. **Home-Based Care**

   - Care by parents, relatives, or nannies.
   - Often more flexible but less standardized.

2. **Daycare Centers**

   - Formal facilities for groups of children.
   - Provide structured schedules, learning activities, and meals.
   - Typically regulated by government authorities.

3. **Preschools/Kindergartens**

   - Early education programs.
   - Combine childcare with educational preparation for primary school.

4. **After-School Programs**

   - Care for school-aged children outside of standard school hours.
   - May include enrichment activities, tutoring, or sports.

---

## ⚖️ Regulatory Compliance

In most countries (and especially in places like the **Netherlands**, where your compliance checker project applies), childcare is strictly regulated to ensure **safety, quality, and staff-to-child ratios**.

Key rules include:

- **Staff-to-child ratios**: e.g., for 15 children, at least 2 staff members must be present.
- **Background checks**: staff must be vetted.
- **Training requirements**: caregivers must hold certifications.
- **Facility standards**: safety, hygiene, fire codes, and space per child.
- **Daily schedules**: balance of meals, play, rest, and education.

Your **automation project** is directly tied to these rules: parsing schedules (from Word, PDF, Excel) to verify if a center is compliant with legal standards.

---

## 🧩 Challenges in Childcare

1. **Staff shortages**: Maintaining required ratios is a global issue.
2. **Compliance complexity**: Rules vary by age group, time of day, and type of activity.
3. **Costs**: Balancing affordability for parents with fair wages for staff.
4. **Digitalization**: Many centers still manage schedules manually (paper or spreadsheets).
5. **Parental trust**: Child safety and well-being are paramount.

---

## 💡 Role of AI & Automation in Childcare

- **Compliance checkers** (like your system): automatically validate staffing against regulations.
- **OCR & document processing**: scan timetables, attendance logs, or government forms.
- **Computer vision**: monitor safety (detect unattended children or unusual activity).
- **Scheduling optimization**: balance staff availability with demand.
- **Parent communication**: chatbots or mobile apps for updates and notifications.

## How to install & use

### Run venv

```bash
.\venv\Scripts\activate.bat
source venv/Scripts/activate
venv\Scripts\activate
myenv\Scripts\activate

py -3.10 -m venv venv
venv\Scripts\activate

py -3.10 -m venv myenv310_01
myenv310_01\Scripts\activate
```

### install EASYOCR

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install easyocr
```

```bash
pip install torch-2.3.0+cu121-cp310-cp310-win_amd64.whl
pip install torchvision-0.18.0+cu121-cp310-cp310-win_amd64.whl
pip install torchaudio-2.3.0+cu121-cp310-cp310-win_amd64.whl

pip install easyocr
```

- paddleocr

```bash
pip install paddlepaddle
pip install paddleocr

# for GPU
pip install paddlepaddle-gpu==2.5.2 -f https://www.paddlepaddle.org.cn/whl/mkl/avx/stable.html

https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html
```

---

```
uvicorn main:app --reload
```

---

read vgc_list in various format(pdf, docx, doc, txt)

```
pip install pymupdf python-docx textract

pip install -r requirements.txt -c constraints.txt
```