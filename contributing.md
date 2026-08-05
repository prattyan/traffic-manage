# Contributing Guidelines

Thank you for your interest in contributing to the **Real-Time Traffic Monitoring & Intelligent Control System** 🚦  
Contributions are welcome and appreciated!

---

## 📌 How to Contribute

You can contribute in the following ways:

- 🐞 Reporting bugs
- ✨ Suggesting new features or enhancements
- 📄 Improving documentation
- 🧠 Enhancing AI models (YOLOv8 / LSTM)
- ⚙️ Optimizing performance or scalability
- 🧪 Adding tests or improving code quality

---

## 🛠️ Getting Started

### 1️⃣ Fork the Repository
Click the **Fork** button on GitHub to create your own copy of the repository.

### 2️⃣ Clone Your Fork
```bash
git clone https://github.com/prattyan/traffic-manage.git
cd traffic-manage
```

### 3️⃣ Create a New Branch
```bash
git checkout -b feature/your-feature-name
```

Use clear and descriptive branch names, such as:
- `feature/emergency-vehicle-detection`
- `fix/dashboard-refresh-bug`
- `docs/readme-update`

---

## 🧪 Development Guidelines

- Follow **PEP 8** coding standards for Python
- Write clean, readable, and well-documented code
- Keep commits focused and meaningful
- Test your changes before submitting
- Avoid breaking existing functionality

---

## 📦 Project Structure Overview

- `app.py` – Main application logic  
- `traffic_lstm.h5` – Pre-trained LSTM model  
- `yolov8n.pt` – YOLOv8 model weights  
- `requirements.txt` – Project dependencies  

Please place new files in appropriate directories and update documentation if needed.

---

## 📝 Commit Message Format

Use clear commit messages:
```text
type: short description

Example:
feat: add emergency vehicle override logic
fix: resolve dashboard update delay
docs: update installation instructions
```

---

## 🔁 Submitting a Pull Request

1. Push your changes to your fork
2. Open a Pull Request (PR) against the `main` branch
3. Provide a clear description of:
   - What was changed
   - Why it was changed
   - Any related issues (if applicable)

Your PR will be reviewed, and feedback may be provided before merging.

---

## 🐛 Reporting Issues

If you find a bug or have a suggestion:
- Open an **Issue** on GitHub
- Include steps to reproduce (if applicable)
- Provide screenshots or logs when helpful

---

## 📜 Code of Conduct

By contributing, you agree to maintain a respectful and inclusive environment.  
Harassment, discrimination, or abusive behavior will not be tolerated.

---

## 🙌 Thank You

Thank you for helping improve this project!  
Your contributions make smart traffic systems more reliable and impactful 🚦
