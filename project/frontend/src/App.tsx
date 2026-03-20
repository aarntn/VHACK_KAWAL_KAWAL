import { useEffect, useState } from 'react';
import './App.css';
import AdminDashboard from './AdminDashboard';
import Dashboard from './Dashboard';
import UserApp from './UserApp';
import DetailUser from './DetailUser';
import './index.css';

export default function App() {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const handleLocationChange = () => {
      setPath(window.location.pathname);
    };

    window.addEventListener('popstate', handleLocationChange);
    return () => window.removeEventListener('popstate', handleLocationChange);
  }, []);

  // Simple routing logic: /dashboard/:id shows transaction details,
  // /dashboard shows the Figma table dashboard,
  // /admin shows the existing admin console, everything else shows the user flow.
  const dashboardIdMatch = path.match(/^\/dashboard\/([^/]+)$/);
  const isDetailUser = !!dashboardIdMatch;
  const isTableDashboard = path === '/dashboard' || path === '/dashboard/';
  const isAdmin = path.startsWith('/admin');
  const transactionId = dashboardIdMatch ? dashboardIdMatch[1] : null;

  return (
    <div className="main-wrapper">
      {isDetailUser && transactionId ? (
        <DetailUser transactionId={transactionId} />
      ) : isTableDashboard ? (
        <Dashboard />
      ) : isAdmin ? (
        <AdminDashboard />
      ) : (
        <UserApp />
      )}
    </div>
  );
}