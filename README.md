# Oral Care AI - Web Application

A healthcare SaaS platform for oral health analysis using AI-powered image processing with advanced spectral imaging capabilities.

## 🌟 Features

### Patient Portal
- **Image Upload**: Upload dental images (JPEG, PNG, WebP) for AI analysis
- **AI Analysis Reports**: View detailed AI-generated dental health reports
- **Health Score**: Overall dental health score with breakdown by category
- **Dentist Connection**: Connect with verified dentists for professional review
- **Chat**: Real-time messaging with connected dentists
- **Report History**: Access all past scan reports with unique Report IDs

### Dentist Portal
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
- **Dentist Review Workflow**:
  - Accept AI results
  - Edit diagnosis with clinical notes
  - Reject AI output with explanation
- **Report Generation**: 
  - Patient information capture (name, phone)
  - Clinical and patient-friendly reports
  - Unique Report ID (SPR-YYYYMMDD-XXXXXXXX)
- **History Tab**:
  - View all past spectral analyses
  - Search by Report ID
  - Download PDF reports with full analysis and images
  - Patient name and phone display

### Admin Dashboard
- **Feature Management**: Enable/disable features system-wide
  - Dentist Dashboard toggle (blocks dentist registration & login)
  - Spectral AI toggle (hides spectral analysis tab)
- **Custom Messages**: Set "coming soon" messages for disabled features
- **Audit Trail**: Track who changed feature settings and when
- **Real-time Updates**: Changes take effect immediately

### Security & Compliance
- JWT authentication with refresh tokens
- Role-based access control (user, dentist, admin)
- Phone numbers stored as salted hashes
- Images encrypted at rest
- Rate limiting on all endpoints
- CORS configuration
- Input validation & sanitization
- HIPAA-aware design

## 🏗️ Project Structure

```
oral-care-ai/
├── frontend/                 # React + TypeScript + Tailwind
│   ├── src/
│   │   ├── components/       # UI components (Dashboard, Cards, etc.)
│   │   ├── pages/            # Route pages
│   │   ├── context/          # Auth context with token persistence
│   │   ├── services/         # API client with auto-refresh
│   │   └── types/            # TypeScript definitions
│   └── tests/                # Vitest & Playwright tests
├── backend-php/              # PHP REST API (XAMPP)
│   ├── api/
│   │   ├── controllers/      # AuthController, SpectralController, etc.
│   │   └── index.php         # API router
│   ├── config/               # Database & app configuration
│   ├── middleware/           # Auth, CORS, Rate limiting
│   ├── services/             # Business logic services
│   └── storage/              # Uploaded images storage
├── ml-model/                 # Python ML Backend
│   ├── api_pytorch.py        # FastAPI server (PyTorch)
│   ├── spectral_dental_pipeline.py  # Spectral analysis & visualization
│   └── models/               # Trained model weights
├── database/                 # MySQL schemas & seeds
├── docs/                     # OpenAPI documentation
└── docker/                   # Docker configurations
```

## 🚀 Quick Start

### Prerequisites
- XAMPP (PHP 8.0+, MySQL 8.0+)
- Node.js 18+
- Python 3.10+ (for ML server)

### 1. Database Setup
```bash
# Import schema and seed data
mysql -u root < database/schema.sql
mysql -u root oral_care_ai < database/seed.sql
```

### 2. Backend PHP Setup
```bash
cd backend-php
cp .env.example .env
# Edit .env with your settings
composer install
```

### 3. ML Server Setup
```bash
cd ml-model
pip install -r requirements.txt
# or for enhanced features:
pip install -r requirements_enhanced.txt

# Start ML API server
python api_pytorch.py
# Runs on http://localhost:8000
```

### 4. Frontend Setup
```bash
cd frontend
cp .env.example .env
npm install
npm run dev
# Runs on http://localhost:5173
```

### 5. Start Services
1. Start XAMPP (Apache + MySQL)
2. Start ML server: `python ml-model/api_pytorch.py`
3. Start frontend: `cd frontend && npm run dev`

## ⚙️ Environment Variables

### Frontend (.env)
```env
VITE_API_URL=http://localhost/oral-care-ai/backend-php/api
```

### Backend PHP (.env)
```env
DB_HOST=localhost
DB_NAME=oral_care_ai
DB_USER=root
DB_PASS=
JWT_SECRET=your-secret-key-min-32-chars
ENCRYPTION_KEY=your-encryption-key-32-chars
USE_REAL_ML=true
ML_API_URL=http://localhost:8000
```

## 👤 Demo Accounts

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@oralcare.ai | password123 |
| Dentist | dr.sarah.chen@dental.com | password123 |
| Patient | john.doe@example.com | password123 |

## 📡 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get tokens
- `POST /auth/refresh` - Refresh access token
- `POST /auth/logout` - Logout and revoke tokens

### Scans
- `GET /scans` - List user's scans
- `POST /scans` - Upload new scan
- `GET /scans/{id}` - Get scan details
- `POST /scans/{id}/analyze` - Run AI analysis
- `GET /scans/{id}/image` - Get scan image

### Spectral Analysis (Dentist Only)
- `POST /spectral/analyze` - Run spectral AI analysis
- `POST /spectral/{id}/review` - Submit dentist review
- `POST /spectral/{id}/report` - Generate report
- `GET /spectral/{id}/report/download` - Download PDF report
- `GET /spectral/history` - Get analysis history

### Connections
- `GET /connections` - List connections
- `POST /connections` - Request connection
- `PATCH /connections/{id}` - Accept/decline connection

### Feature Flags (Admin)
- `GET /features` - Get all feature flags (public)
- `GET /features/admin` - Get detailed feature info (admin only)
- `PATCH /features/{key}` - Update feature flag (admin only)
- `GET /features/check/{key}` - Check if feature is enabled

## 🧪 Running Tests

```bash
# Frontend unit tests
cd frontend && npm test

# Frontend E2E tests
cd frontend && npm run test:e2e

# PHP backend tests
cd backend-php && ./vendor/bin/phpunit
```

## 📊 ML Model Details

The spectral analysis uses a multi-stage pipeline:
1. **Preprocessing**: Image normalization, noise reduction
2. **Feature Extraction**: CNN-based feature extraction
3. **Spectral Analysis**: PCA for dimensionality reduction
4. **Classification**: Ensemble classifier for condition detection
5. **Visualization**: Color-coded segmentation overlay generation

Supported image types:
- **NIR (Near-Infrared)**: Best for subsurface decay detection
- **Fluorescence**: Optimal for early caries and bacterial detection
- **Intraoral Camera**: Standard dental photography analysis

## 📝 License

MIT License - See LICENSE file for details.
