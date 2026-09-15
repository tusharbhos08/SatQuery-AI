import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { DashboardView } from './components/DashboardView';
import { AnalyzeView } from './components/AnalyzeView';
import { ChangeDetectionView } from './components/ChangeDetectionView';
import { OpticalSarView } from './components/OpticalSarView';
import { WaterFinderView } from './components/WaterFinderView';
import { DemoModeView } from './components/DemoModeView';
import { ModelsView } from './components/ModelsView';
import { HistoryView } from './components/HistoryView';

export function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');

  const getPageInfo = () => {
    switch (currentTab) {
      case 'dashboard':
        return { title: 'Dashboard & Mission Control', subtitle: 'Real-Time Earth Observation AI Intelligence' };
      case 'analyze':
        return { title: 'Multimodal Remote-Sensing Analysis', subtitle: 'Single-Image VQA, Grounding, NDVI, & Multi-Step Reasoning' };
      case 'change':
        return { title: 'Bi-Temporal Change Understanding', subtitle: 'Co-Registered Multi-Date Surface Dynamics & Classification' };
      case 'optical-sar':
        return { title: 'Optical + SAR Feature-Level Fusion', subtitle: 'Cross-Modal Spectral + Radar Consensus Mapping' };
      case 'water-finder':
        return { title: 'Automated Water Body Finder', subtitle: 'Morphological Segmentation & Surface Area Computation' };
      case 'demo':
        return { title: 'SIH Final Demonstration Showcase', subtitle: '7 Live Evaluator Workflows Powered by Real ML Models' };
      case 'models':
        return { title: 'Model Registry & Domain Adaptation Lab', subtitle: 'BigEarthNet Fine-Tuning & Architecture Specifications' };
      case 'history':
        return { title: 'Observable Execution History', subtitle: 'Full Audit Trail of AI Agent Inferences & Downloads' };
      default:
        return { title: 'SatQuery AI', subtitle: 'Ask Questions. Understand Earth.' };
    }
  };

  const { title, subtitle } = getPageInfo();

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#0B0F19' }}>
      {/* Sidebar */}
      <Sidebar currentTab={currentTab} onTabChange={setCurrentTab} />

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Navbar title={title} subtitle={subtitle} />
        
        <main style={{ flex: 1, overflowY: 'auto' }}>
          {currentTab === 'dashboard' && <DashboardView onNavigate={setCurrentTab} />}
          {currentTab === 'analyze' && <AnalyzeView />}
          {currentTab === 'change' && <ChangeDetectionView />}
          {currentTab === 'optical-sar' && <OpticalSarView />}
          {currentTab === 'water-finder' && <WaterFinderView />}
          {currentTab === 'demo' && <DemoModeView />}
          {currentTab === 'models' && <ModelsView />}
          {currentTab === 'history' && <HistoryView />}
        </main>
      </div>
    </div>
  );
}

export default App;
