import { lazy, Suspense, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const App = lazy(() => import('./App'));
const IndexingPage = lazy(() => import('./indexing/IndexingPage'));
const Bm25Page = lazy(() => import('./bm25/Bm25Page'));

const route = () => (location.hash.startsWith('#/indexing') ? 'indexing' : location.hash.startsWith('#/bm25') ? 'bm25' : 'app');

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
  return <Suspense fallback={null}>{page === 'indexing' ? <IndexingPage /> : page === 'bm25' ? <Bm25Page /> : <App />}</Suspense>;
}

createRoot(document.getElementById('root')!).render(<Root />);
