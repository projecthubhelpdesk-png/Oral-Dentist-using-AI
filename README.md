# 🦷 Oral Care AI - Intelligent Dental Health Platform

<div align="center">

![Oral Care AI](https://img.shields.io/badge/Oral%20Care-AI-blue?style=for-the-badge&logo=tooth)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-orange?style=for-the-badge)

**A comprehensive healthcare SaaS platform for oral health analysis using AI-powered image processing with advanced spectral imaging capabilities.**

[Features](#-features) • [Tech Stack](#-tech-stack) • [Installation](#-installation) • [API Docs](#-api-endpoints) • [Demo](#-demo-accounts)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Demo Accounts](#-demo-accounts)
- [API Endpoints](#-api-endpoints)
- [ML Model Details](#-ml-model-details)
- [Testing](#-running-tests)
- [License](#-license)

---

## 🌟 Features

### � Patienta Portal
- **Image Upload**: Upload dental images (JPEG, PNG, WebP) for AI analysis
- **AI Analysis Reports**: View detailed AI-generated dental health reports
- **Health Score**: Overall dental health score with breakdown by category
- **Dentist Connection**: Connect with verified dentists for professional review
- **Real-time Chat**: Messaging with connected dentists
- **Report History**: Access all past scan reports with unique Report IDs

### 🩺 Dentist Portal
- **Dashboard Overview**: Statistics on patients, consultations, procedures
- **Patient Scans**: Review patient-uploaded scans with AI analysis
- **AI-Flagged Cases**: Urgent, attention-needed, and normal case categorization
- **Professional Assessment**: Accept, edit, or reject AI findings
- **Report Search**: Search patient reports by Report ID

### 🔬 Spectral AI Analysis (Advanced)
- **Multi-Spectral Imaging**: Support for NIR (Near-Infrared), Fluorescence, and Intraoral camera images
- **Advanced Detection**:
  - Early caries (white spot lesions)
  - Enamel demineralization
  - Subsurface decay
  - Gingival inflammation
  - Dental calculus
  - Periodontal issues
- **Spectral Visualization**: Color-coded segmentation overlay showing:
  - 🟢 Healthy Enamel (Green)
  - 🟢 Healthy Gingiva (Dark Green)
  - 🔴 Caries/Decay (Red)
  - 🟠 Early Caries (Orange)
  - 🔵 Calculus (Cyan)
  - 🟣 Inflammation (Purple)
  - 💜 Demineralization (Magenta)
- **Side-by-Side Comparison**: Original image vs spectral analysis visualization
- **Health Score**: CNN + PCA + Ensemble Classifier based scoring
- **Dentist Review Workflow**: Accept, Edit, or Reject AI results
- **PDF Report Generation**: Full analysis with images, patient info, clinical notes
- **History Tab**: View past analyses, search by Report ID, download reports

### 🛡️ Admin Dashboard
- **Feature Management**: Enable/disable features system-wide
  - Dentist Dashboard toggle (blocks registration & login)
  - Spectral AI toggle (hides spectral analysis tab)
- **Custom Messages**: Set "coming soon" messages for disabled features
- **Audit Trail**: Track feature changes with timestamps
- **Real-time Updates**: Changes take effect immediately

### 🔒 Security & Compliance
- JWT authentication with refresh tokens
- Role-based access control (user, dentist, admin)
- Phone numbers stored as salted hashes
- Images encrypted at rest
- Rate limiting on all endpoints
- CORS configuration
- Input validation & sanitization
- HIPAA-aware design

---

## 🛠️ Tech Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| ![React](https://img.shields.io/badge/React-18.2.0-61DAFB?logo=react) | 18.2.0 | UI Framework |
| ![TypeScript](https://img.shields.io/badge/TypeScript-5.2.2-3178C6?logo=typescript) | 5.2.2 | Type Safety |
| ![Vite](https://img.shields.io/badge/Vite-5.0.0-646CFF?logo=vite) | 5.0.0 | Build Tool |
| ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.3.5-06B6D4?logo=tailwindcss) | 3.3.5 | Styling |
| ![React Router](https://img.shields.io/badge/React_Router-6.20.0-CA4245?logo=reactrouter) | 6.20.0 | Routing |
| ![Axios](https://img.shields.io/badge/Axios-1.6.2-5A29E4?logo=axios) | 1.6.2 | HTTP Client |
| ![Zustand](https://img.shields.io/badge/Zustand-4.4.7-orange) | 4.4.7 | State Management |

#### Frontend Dev Dependencies
- **Vitest** - Unit Testing Framework
- **Playwright** - E2E Testing
- **ESLint** - Code Linting
- **PostCSS** - CSS Processing
- **Autoprefixer** - CSS Vendor Prefixes

### Backend (PHP)
| Technology | Version | Purpose |
|------------|---------|---------|
| ![PHP](https://img.shields.io/badge/PHP-8.1+-777BB4?logo=php) | 8.1+ | Server Language |
| ![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?logo=mysql) | 8.0 | Database |
| ![JWT](https://img.shields.io/badge/Firebase_JWT-6.9-FFCA28?logo=firebase) | 6.9 | Authentication |
| ![XAMPP](https://img.shields.io/badge/XAMPP-Apache-FB7A24?logo=xampp) | Latest | Local Server |
| ![PHPUnit](https://img.shields.io/badge/PHPUnit-10.4-3C9CD7) | 10.4 | Testing |

### Machine Learning (Python)
| Technology | Version | Purpose |
|------------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python) | 3.10+ | ML Language |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?logo=tensorflow) | 2.13+ | Deep Learning |
| ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi) | 0.100+ | API Framework |
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?logo=opencv) | 4.5+ | Image Processing |
| ![Ultralytics](https://img.shields.io/badge/YOLOv8-8.0+-00FFFF) | 8.0+ | Object Detection |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikitlearn) | 1.0+ | ML Utilities |
| ![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?logo=numpy) | 1.21+ | Numerical Computing |
| ![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458?logo=pandas) | 1.3+ | Data Processing |
| ![Pillow](https://img.shields.io/badge/Pillow-9.0+-green) | 9.0+ | Image Handling |
| ![Uvicorn](https://img.shields.io/badge/Uvicorn-0.20+-purple) | 0.20+ | ASGI Server |

#### ML Additional Libraries
- **Matplotlib** & **Seaborn** - Data Visualization
- **tifffile** & **imagecodecs** - Spectral TIFF Support
- **OpenAI SDK** - LLM Integration (Optional)
- **Pydantic** - Data Validation
- **aiohttp** & **httpx** - Async HTTP

### DevOps & Tools
| Technology | Purpose |
|------------|---------|
| ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) | Containerization |
| ![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white) | Version Control |
| ![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white) | Repository |
| ![VS Code](https://img.shields.io/badge/VS_Code-007ACC?logo=visualstudiocode&logoColor=white) | IDE |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              React + TypeScript + Tailwind               │   │
│  │         (Vite Dev Server - localhost:5173)               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER                               │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │    PHP REST API      │    │     Python ML API            │  │
│  │  (localhost:8080)    │◄──►│   (FastAPI - localhost:8000) │  │
│  │                      │    │                              │  │
│  │  • Authentication    │    │  • TensorFlow Models         │  │
│  │  • User Management   │    │  • YOLOv8 Detection          │  │
│  │  • Scan Management   │    │  • Spectral Analysis         │  │
│  │  • Feature Flags     │    │  • Image Processing          │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │      MySQL 8.0       │    │     File Storage             │  │
│  │                      │    │                              │  │
│  │  • Users             │    │  • Uploaded Scans            │  │
│  │  • Scans             │    │  • Spectral Images           │  │
│  │  • Reports           │    │  • ML Model Weights          │  │
│  │  • Feature Flags     │    │  • Generated Reports         │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
oral-care-ai/
├── 📂 frontend/                  # React Frontend Application
│   ├── 📂 src/
│   │   ├── 📂 components/        # Reusable UI Components
│   │   │   ├── 📂 dashboard/     # Dashboard Components
│   │   │   └── 📂 ui/            # Base UI Components (Button, Card, etc.)
│   │   ├── 📂 pages/             # Route Pages
│   │   ├── 📂 context/           # React Context (Auth)
│   │   ├── 📂 services/          # API Services
│   │   ├── 📂 hooks/             # Custom React Hooks
│   │   └── 📂 types/             # TypeScript Definitions
│   ├── 📂 tests/                 # Test Files
│   ├── 📄 package.json
│   ├── 📄 vite.config.ts
│   ├── 📄 tailwind.config.js
│   └── 📄 tsconfig.json
│
├── 📂 backend-php/               # PHP REST API
│   ├── 📂 api/
│   │   ├── 📂 controllers/       # API Controllers
│   │   │   ├── 📄 AuthController.php
│   │   │   ├── 📄 ScanController.php
│   │   │   ├── 📄 SpectralController.php
│   │   │   └── 📄 FeatureController.php
│   │   └── 📄 index.php          # API Router
│   ├── 📂 config/                # Configuration Files
│   ├── 📂 middleware/            # Auth, CORS, Rate Limiting
│   ├── 📂 services/              # Business Logic
│   ├── 📂 models/                # Data Models
│   ├── 📂 storage/               # File Storage
│   ├── 📄 composer.json
│   └── 📄 .env
│
├── 📂 ml-model/                  # Python ML Backend
│   ├── 📄 api_pytorch.py         # FastAPI Server
│   ├── 📄 spectral_dental_pipeline.py  # Spectral Analysis
│   ├── 📄 advanced_dental_pipeline.py  # Advanced Pipeline
│   ├── 📂 models/                # Trained Model Weights
│   ├── 📂 datasets/              # Training Datasets
│   └── 📄 requirements_enhanced.txt
│
├── 📂 database/                  # Database Files
│   ├── 📄 schema.sql             # Database Schema
│   └── 📄 seed.sql               # Seed Data
│
├── 📂 docker/                    # Docker Configuration
│   ├── 📄 docker-compose.yml
│   ├── 📄 Dockerfile.frontend
│   └── 📄 Dockerfile.php
│
├── 📂 docs/                      # Documentation
│   └── 📄 openapi.yaml           # API Documentation
│
├── 📄 start.bat                  # Start Frontend + Backend
├── 📄 start-all.bat              # Start All Services (incl. ML)
├── 📄 stop.bat                   # Stop All Services
└── 📄 README.md
```

---

## 🚀 Installation

### Prerequisites
- **XAMPP** (PHP 8.1+, MySQL 8.0+, Apache)
- **Node.js** 18+ and npm
- **Python** 3.10+
- **Git**

### Quick Start (Windows)

```bash
# Clone the repository
git clone https://github.com/projecthubhelpdesk-png/Oral-Dentist-using-AI.git
cd Oral-Dentist-using-AI

# Run all services with one command
start-all.bat
```

### Manual Setup

#### 1. Database Setup
```bash
# Start MySQL from XAMPP
# Then import schema and seed data
mysql -u root < database/schema.sql
mysql -u root oral_care_ai < database/seed.sql
```

#### 2. Backend PHP Setup
```bash
cd backend-php
copy .env.example .env
# Edit .env with your database credentials
composer install
```

#### 3. ML Server Setup
```bash
cd ml-model
pip install -r requirements_enhanced.txt

# Start ML API server
python api_pytorch.py
# Runs on http://localhost:8000
```

#### 4. Frontend Setup
```bash
cd frontend
copy .env.example .env
npm install
npm run dev
# Runs on http://localhost:5173
```

### Using Batch Files

| File | Command | Description |
|------|---------|-------------|
| `start.bat` | Double-click | Starts Frontend + PHP Backend |
| `start-all.bat` | Double-click | Starts Frontend + PHP Backend + ML API |
| `stop.bat` | Double-click | Stops all running services |

---

## ⚙️ Environment Variables

### Frontend (`frontend/.env`)
```env
VITE_API_URL=http://localhost:8080/api
```

### Backend PHP (`backend-php/.env`)
```env
# Database
DB_HOST=localhost
DB_NAME=oral_care_ai
DB_USER=root
DB_PASS=

# Security
JWT_SECRET=your-secret-key-minimum-32-characters
ENCRYPTION_KEY=your-encryption-key-32-characters

# ML Integration
USE_REAL_ML=true
ML_API_URL=http://localhost:8000
```

---

## 👤 Demo Accounts

| Role | Email | Password |
|------|-------|----------|
| 🛡️ Admin | admin@oralcare.ai | password123 |
| 🩺 Dentist | dr.sarah.chen@dental.com | password123 |
| 👤 Patient | john.doe@example.com | password123 |

---

## 📡 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login and get tokens |
| POST | `/auth/refresh` | Refresh access token |
| POST | `/auth/logout` | Logout and revoke tokens |

### Scans
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/scans` | List user's scans |
| POST | `/scans` | Upload new scan |
| GET | `/scans/{id}` | Get scan details |
| POST | `/scans/{id}/analyze` | Run AI analysis |
| GET | `/scans/{id}/image` | Get scan image |

### Spectral Analysis (Dentist Only)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/spectral/analyze` | Run spectral AI analysis |
| POST | `/spectral/{id}/review` | Submit dentist review |
| POST | `/spectral/{id}/report` | Generate report |
| GET | `/spectral/{id}/report/download` | Download PDF report |
| GET | `/spectral/history` | Get analysis history |

### Feature Flags
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/features` | Get all feature flags (public) |
| GET | `/features/admin` | Get detailed info (admin only) |
| PATCH | `/features/{key}` | Update feature (admin only) |
| GET | `/features/check/{key}` | Check if feature enabled |

### Connections
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/connections` | List connections |
| POST | `/connections` | Request connection |
| PATCH | `/connections/{id}` | Accept/decline |

---

## 🧠 ML Model Details

### Pipeline Architecture
```
Input Image
     │
     ▼
┌─────────────────┐
│  Preprocessing  │  ← Normalization, Noise Reduction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Extract │  ← CNN (EfficientNet-B4, ResNet50)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Spectral Anal.  │  ← PCA Dimensionality Reduction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classification  │  ← Ensemble Classifier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │  ← Color-coded Segmentation
└─────────────────┘
```

### Supported Image Types
| Type | Best For |
|------|----------|
| **NIR (Near-Infrared)** | Subsurface decay detection |
| **Fluorescence** | Early caries, bacterial detection |
| **Intraoral Camera** | Standard dental photography |

### Detection Capabilities
- Early caries (white spot lesions)
- Enamel demineralization
- Subsurface decay
- Gingival inflammation
- Dental calculus
- Periodontal issues

---

## 🧪 Running Tests

```bash
# Frontend unit tests
cd frontend
npm test

# Frontend E2E tests
cd frontend
npm run test:e2e

# PHP backend tests
cd backend-php
./vendor/bin/phpunit

# Python ML tests
cd ml-model
pytest
```

---

## 🐳 Docker Deployment

```bash
cd docker
docker-compose up -d
```

Services:
- Frontend: http://localhost:3000
- Backend: http://localhost:8080
- ML API: http://localhost:8000
- MySQL: localhost:3306

---

## 📊 Languages & Code Distribution

| Language | Usage |
|----------|-------|
| TypeScript/JavaScript | Frontend (React) |
| PHP | Backend REST API |
| Python | Machine Learning |
| SQL | Database |
| HTML/CSS | UI Templates |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for Better Dental Health**

[⬆ Back to Top](#-oral-care-ai---intelligent-dental-health-platform)

</div>
