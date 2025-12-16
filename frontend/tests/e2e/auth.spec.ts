import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('should display login page', async ({ page }) => {
    await page.goto('/login');
    
    await expect(page.getByRole('heading', { name: /sign in/i })).toBeVisible();
    await expect(page.getByLabel(/email/i)).toBeVisible();
    await expect(page.getByLabel(/password/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');
    
    await page.getByLabel(/email/i).fill('invalid@example.com');
    await page.getByLabel(/password/i).fill('wrongpassword');
    await page.getByRole('button', { name: /sign in/i }).click();
    
    await expect(page.getByText(/invalid/i)).toBeVisible();
  });

  test('should navigate to register page', async ({ page }) => {
    await page.goto('/login');
    
    await page.getByRole('link', { name: /sign up/i }).click();
    
    await expect(page).toHaveURL('/register');
    await expect(page.getByRole('heading', { name: /create your account/i })).toBeVisible();
  });

  test('should validate registration form', async ({ page }) => {
    await page.goto('/register');
    
    // Try to submit empty form
    await page.getByRole('button', { name: /create account/i }).click();
    
    // Check for validation
    await expect(page.getByLabel(/email/i)).toHaveAttribute('required');
    await expect(page.getByLabel(/^password$/i)).toHaveAttribute('required');
  });

  test('should show password mismatch error', async ({ page }) => {
    await page.goto('/register');
    
    await page.getByLabel(/email/i).fill('test@example.com');
    await page.getByLabel(/^password$/i).fill('password123');
    await page.getByLabel(/confirm password/i).fill('different123');
    
    // Check terms checkbox
    await page.getByRole('checkbox').check();
    
    await page.getByRole('button', { name: /create account/i }).click();
    
    await expect(page.getByText(/passwords do not match/i)).toBeVisible();
  });

  test('should allow role selection on register', async ({ page }) => {
    await page.goto('/register');
    
    // Default should be patient
    const patientButton = page.getByRole('button', { name: /patient/i });
    await expect(patientButton).toHaveClass(/border-primary-500/);
    
    // Click dentist
    await page.getByRole('button', { name: /dentist/i }).click();
    await expect(page.getByRole('button', { name: /dentist/i })).toHaveClass(/border-primary-500/);
  });
});

test.describe('Protected Routes', () => {
  test('should redirect to login when not authenticated', async ({ page }) => {
    await page.goto('/dashboard');
    
    await expect(page).toHaveURL('/login');
  });

  test('should redirect to login when accessing scans', async ({ page }) => {
    await page.goto('/scans');
    
    await expect(page).toHaveURL('/login');
  });
});
