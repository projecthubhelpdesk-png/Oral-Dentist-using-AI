import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import * as express from 'express';
import * as cors from 'cors';

admin.initializeApp();

const db = admin.firestore();
const storage = admin.storage();

const app = express();
app.use(cors({ origin: true }));
app.use(express.json());

// ============================================
// AUTH TRIGGERS
// ============================================

/**
 * Create user document when a new user signs up
 */
export const onUserCreate = functions.auth.user().onCreate(async (user) => {
  const userData = {
    id: user.uid,
    email: user.email,
    role: 'user', // Default role, can be updated
    emailVerified: user.emailVerified,
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
    isActive: true,
  };

  await db.collection('users').doc(user.uid).set(userData);

  // Create audit log
  await db.collection('auditLogs').add({
    userId: user.uid,
    action: 'USER_CREATED',
    resourceType: 'user',
    resourceId: user.uid,
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
  });
});

/**
 * Clean up user data when account is deleted
 */
export const onUserDelete = functions.auth.user().onDelete(async (user) => {
  const batch = db.batch();

  // Mark user as inactive (soft delete)
  batch.update(db.collection('users').doc(user.uid), {
    isActive: false,
    deletedAt: admin.firestore.FieldValue.serverTimestamp(),
  });

  // Archive all user's scans
  const scans = await db.collection('scans').where('userId', '==', user.uid).get();
  scans.forEach((doc) => {
    batch.update(doc.ref, { status: 'archived' });
  });

  await batch.commit();

  // Audit log
  await db.collection('auditLogs').add({
    userId: user.uid,
    action: 'USER_DELETED',
    resourceType: 'user',
    resourceId: user.uid,
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
  });
});

// ============================================
// STORAGE TRIGGERS
// ============================================

/**
 * Process uploaded scan image
 */
export const onScanUpload = functions.storage.object().onFinalize(async (object) => {
  const filePath = object.name;
  if (!filePath || !filePath.startsWith('scans/')) {
    return;
  }

  // Extract userId and scanId from path: scans/{userId}/{scanId}/{filename}
  const pathParts = filePath.split('/');
  if (pathParts.length < 4) return;

  const userId = pathParts[1];
  const scanId = pathParts[2];

  // Update scan document with storage info
  await db.collection('scans').doc(scanId).update({
    imageStoragePath: filePath,
    fileSize: parseInt(object.size || '0', 10),
    mimeType: object.contentType,
    imageHash: object.md5Hash,
    status: 'uploaded',
    uploadedAt: admin.firestore.FieldValue.serverTimestamp(),
  });

  // TODO: Generate thumbnail
  // TODO: Trigger AI analysis

  // Audit log
  await db.collection('auditLogs').add({
    userId,
    action: 'SCAN_UPLOADED',
    resourceType: 'scan',
    resourceId: scanId,
    details: { filePath, size: object.size },
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
  });
});

// ============================================
// CALLABLE FUNCTIONS
// ============================================

/**
 * Trigger AI analysis for a scan
 */
export const analyzeScan = functions.https.onCall(async (data, context) => {
  if (!context.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Must be logged in');
  }

  const { scanId } = data;
  if (!scanId) {
    throw new functions.https.HttpsError('invalid-argument', 'scanId is required');
  }

  // Get scan document
  const scanDoc = await db.collection('scans').doc(scanId).get();
  if (!scanDoc.exists) {
    throw new functions.https.HttpsError('not-found', 'Scan not found');
  }

  const scan = scanDoc.data()!;

  // Verify ownership
  if (scan.userId !== context.auth.uid) {
    throw new functions.https.HttpsError('permission-denied', 'Not authorized');
  }

  // Check if already processed
  if (scan.status !== 'uploaded') {
    throw new functions.https.HttpsError('failed-precondition', 'Scan already processed');
  }

  // Update status to processing
  await scanDoc.ref.update({ status: 'processing' });

  // TODO: Call AI service
  // For now, create mock analysis result
  const mockAnalysis = {
    scanId,
    modelType: scan.scanType,
    modelVersion: scan.scanType === 'basic_rgb' ? '2.1.0' : '1.5.2',
    overallScore: Math.random() * 40 + 60, // 60-100
    confidenceScore: Math.random() * 0.2 + 0.8, // 0.8-1.0
    findings: [
      {
        type: 'healthy',
        severity: 'none',
        location: 'overall',
        confidence: 0.9,
      },
    ],
    riskAreas: [],
    recommendations: ['Continue current oral hygiene routine', 'Schedule regular checkups'],
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
  };

  // Save analysis result
  const analysisRef = await db.collection('analysisResults').add(mockAnalysis);

  // Update scan status
  await scanDoc.ref.update({
    status: 'analyzed',
    processedAt: admin.firestore.FieldValue.serverTimestamp(),
  });

  // Create notification
  await db.collection('notifications').add({
    userId: context.auth.uid,
    type: 'analysis_complete',
    title: 'Scan Analysis Ready',
    message: 'Your dental scan has been analyzed. View your results now.',
    data: { scanId, analysisId: analysisRef.id },
    isRead: false,
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
  });

  return { analysisId: analysisRef.id };
});

/**
 * Create dentist profile
 */
export const createDentistProfile = functions.https.onCall(async (data, context) => {
  if (!context.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'Must be logged in');
  }

  const { licenseNumber, licenseState, specialty } = data;

  // Update user role to dentist
  await db.collection('users').doc(context.auth.uid).update({ role: 'dentist' });

  // Create dentist profile
  const profile = {
    userId: context.auth.uid,
    licenseNumberEncrypted: licenseNumber, // TODO: Encrypt
    licenseState,
    licenseVerified: false,
    specialty: specialty || 'General Dentistry',
    acceptingPatients: true,
    consultationFeeCents: 0,
    averageRating: 0,
    totalReviews: 0,
    createdAt: admin.firestore.FieldValue.serverTimestamp(),
  };

  const profileRef = await db.collection('dentistProfiles').add(profile);

  return { profileId: profileRef.id };
});

// ============================================
// HTTP API (Express)
// ============================================

// Middleware to verify Firebase Auth token
const authenticate = async (
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
) => {
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  const token = authHeader.split('Bearer ')[1];
  try {
    const decodedToken = await admin.auth().verifyIdToken(token);
    (req as any).user = decodedToken;
    next();
  } catch {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Get current user
app.get('/users/me', authenticate, async (req, res) => {
  const userId = (req as any).user.uid;
  const userDoc = await db.collection('users').doc(userId).get();

  if (!userDoc.exists) {
    res.status(404).json({ error: 'User not found' });
    return;
  }

  res.json(userDoc.data());
});

// List scans
app.get('/scans', authenticate, async (req, res) => {
  const userId = (req as any).user.uid;
  const { status, scanType, limit = '20', offset = '0' } = req.query;

  let query: admin.firestore.Query = db
    .collection('scans')
    .where('userId', '==', userId)
    .orderBy('uploadedAt', 'desc');

  if (status) {
    query = query.where('status', '==', status);
  }
  if (scanType) {
    query = query.where('scanType', '==', scanType);
  }

  const snapshot = await query.limit(parseInt(limit as string, 10)).get();

  const scans = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));

  res.json({
    data: scans,
    total: scans.length,
    limit: parseInt(limit as string, 10),
    offset: parseInt(offset as string, 10),
  });
});

// Get analysis for scan
app.get('/scans/:scanId/analysis', authenticate, async (req, res) => {
  const { scanId } = req.params;
  const userId = (req as any).user.uid;

  // Verify scan ownership
  const scanDoc = await db.collection('scans').doc(scanId).get();
  if (!scanDoc.exists || scanDoc.data()?.userId !== userId) {
    res.status(404).json({ error: 'Scan not found' });
    return;
  }

  // Get latest analysis
  const analysisSnapshot = await db
    .collection('analysisResults')
    .where('scanId', '==', scanId)
    .orderBy('createdAt', 'desc')
    .limit(1)
    .get();

  if (analysisSnapshot.empty) {
    res.status(404).json({ error: 'No analysis found' });
    return;
  }

  const analysis = analysisSnapshot.docs[0];
  res.json({ id: analysis.id, ...analysis.data() });
});

// Export Express app as Cloud Function
export const api = functions.https.onRequest(app);
