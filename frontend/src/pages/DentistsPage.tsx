import { useEffect, useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Layout } from '@/components/layout/Layout';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { getDentists } from '@/services/dentists';
import { createConnection } from '@/services/connections';
import type { DentistProfile } from '@/types';

export function DentistsPage() {
  const { user } = useAuth();
  
  // Dentists shouldn't access this page - redirect to dashboard
  if (user?.role === 'dentist') {
    return <Navigate to="/dashboard" replace />;
  }
  const [dentists, setDentists] = useState<(DentistProfile & { email?: string })[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [specialty, setSpecialty] = useState('');
  const [connectingId, setConnectingId] = useState<string | null>(null);

  useEffect(() => {
    loadDentists();
  }, [specialty]);

  async function loadDentists() {
    try {
      setIsLoading(true);
      const response = await getDentists({ 
        specialty: specialty || undefined,
        acceptingPatients: true,
        limit: 50 
      });
      // Filter out current user if they're a dentist
      const filtered = response.data.filter(d => d.userId !== user?.id);
      setDentists(filtered);
    } catch (err) {
      console.error('Failed to load dentists:', err);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleConnect(dentistUserId: string) {
    try {
      setConnectingId(dentistUserId);
      await createConnection(dentistUserId, true);
      alert('Connection request sent!');
    } catch (err: any) {
      alert(err.response?.data?.message || 'Failed to send request');
    } finally {
      setConnectingId(null);
    }
  }

  const specialties = [
    'General Dentistry',
    'Cosmetic Dentistry',
    'Periodontics',
    'Orthodontics',
    'Endodontics',
    'Oral Surgery',
  ];


  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Find a Dentist</h1>
          <p className="text-gray-600">Connect with verified dental professionals for expert reviews</p>
        </div>

        {/* Filters */}
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setSpecialty('')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              !specialty ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            All Specialties
          </button>
          {specialties.map((s) => (
            <button
              key={s}
              onClick={() => setSpecialty(s)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                specialty === s ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Dentists Grid */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-gray-100 rounded-xl h-64 animate-pulse" />
            ))}
          </div>
        ) : dentists.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {dentists.map((dentist) => (
              <Card key={dentist.id} className="flex flex-col">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg">Dr. {(dentist as any).name || dentist.email?.split('@')[0] || 'Dentist'}</CardTitle>
                      <p className="text-sm text-gray-500">{dentist.specialty}</p>
                      {dentist.clinicName && <p className="text-xs text-gray-400">{dentist.clinicName}</p>}
                    </div>
                    {dentist.licenseVerified && (
                      <Badge variant="success">Verified</Badge>
                    )}
                  </div>
                </CardHeader>
                
                <div className="flex-1 space-y-3 text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-yellow-500">★</span>
                    <span className="font-medium">{dentist.averageRating.toFixed(1)}</span>
                    <span className="text-gray-500">({dentist.totalReviews} reviews)</span>
                  </div>
                  
                  <p className="text-gray-600 line-clamp-2">{dentist.bio || 'No bio available'}</p>
                  
                  <div className="flex justify-between text-gray-500">
                    <span>{dentist.yearsExperience} years exp.</span>
                    <span>{dentist.licenseState}</span>
                  </div>
                  
                  {dentist.consultationFeeCents > 0 && (
                    <p className="text-gray-600">
                      Consultation: ${(dentist.consultationFeeCents / 100).toFixed(2)}
                    </p>
                  )}
                </div>

                <div className="mt-4 pt-4 border-t flex gap-2">
                  <Link to={`/dentists/${dentist.userId}`} className="flex-1">
                    <Button variant="secondary" className="w-full">View Profile</Button>
                  </Link>
                  <Button 
                    className="flex-1"
                    onClick={() => handleConnect(dentist.userId)}
                    disabled={connectingId === dentist.userId}
                  >
                    {connectingId === dentist.userId ? 'Sending...' : 'Connect'}
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <Card className="text-center py-12">
            <p className="text-gray-500">No dentists found matching your criteria.</p>
          </Card>
        )}
      </div>
    </Layout>
  );
}
