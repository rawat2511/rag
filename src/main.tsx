import { lazy, Suspense, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const App = lazy(() => import('./App'));
const IndexingPage = lazy(() => import('./indexing/IndexingPage'));

const route = () => (location.hash.startsWith('#/indexing') ? 'indexing' : 'app');

function Root() {
  const [page, setPage] = useState(route);
  useEffect(() => {
    const onHash = () => setPage((prev) => {
      const next = route();
      if (next !== prev) window.scrollTo(0, 0);
      return next;
    });
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return <Suspense fallback={null}>{page === 'indexing' ? <IndexingPage /> : <App />}</Suspense>;
}

createRoot(document.getElementById('root')!).render(<Root />);
