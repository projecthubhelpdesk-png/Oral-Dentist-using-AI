import { useAuth } from '@/context/AuthContext';
import { UserDashboard } from '@/components/dashboard/UserDashboard';
import { DentistDashboardPro } from '@/components/dashboard/DentistDashboardPro';
import { AdminDashboard } from '@/components/dashboard/AdminDashboard';
import { Layout } from '@/components/layout/Layout';

export function DashboardPage() {
  const { user } = useAuth();

  const renderDashboard = () => {
    switch (user?.role) {
      case 'admin':
        return <AdminDashboard />;
      case 'dentist':
        return <DentistDashboardPro />;
      default:
        return <UserDashboard />;
    }
  };

  return <Layout>{renderDashboard()}</Layout>;
}
