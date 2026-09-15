import React from 'react';
import { 
  LayoutDashboard, 
  Search, 
  GitCompare, 
  Layers, 
  Droplets, 
  Cpu, 
  Sparkles, 
  History, 
  FileText 
} from 'lucide-react';

interface SidebarProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentTab, onTabChange }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'analyze', label: 'Analyze (VQA / Grounding)', icon: Search },
    { id: 'change', label: 'Bi-Temporal Change', icon: GitCompare },
    { id: 'optical-sar', label: 'Optical + SAR Fusion', icon: Layers },
    { id: 'water-finder', label: 'Water Body Finder', icon: Droplets },
    { id: 'demo', label: 'SIH Demo Mode', icon: Sparkles, highlight: true },
    { id: 'models', label: 'Model Registry & Lab', icon: Cpu },
    { id: 'history', label: 'Execution History', icon: History },
  ];

  return (
    <aside style={{
      width: '260px',
      backgroundColor: '#0F172A',
      borderRight: '1px solid #1E293B',
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100vh'
    }}>
      {/* Brand Header */}
      <div style={{ padding: '24px 20px', borderBottom: '1px solid #1E293B' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 12px rgba(6, 182, 212, 0.4)'
          }}>
            <Sparkles size={20} color="#FFFFFF" />
          </div>
          <div>
            <h1 style={{ fontSize: '18px', fontWeight: 800, letterSpacing: '-0.5px', color: '#F8FAFC' }}>
              SATQUERY <span style={{ color: '#38BDF8' }}>AI</span>
            </h1>
            <p style={{ fontSize: '10px', color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              SIH26167 Remote Sensing
            </p>
          </div>
        </div>
      </div>

      {/* Navigation List */}
      <nav style={{ padding: '16px 12px', flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '10px 14px',
                borderRadius: '8px',
                border: isActive ? '1px solid #3B82F6' : '1px solid transparent',
                backgroundColor: isActive 
                  ? 'rgba(59, 130, 246, 0.15)' 
                  : item.highlight 
                    ? 'rgba(6, 182, 212, 0.08)' 
                    : 'transparent',
                color: isActive ? '#38BDF8' : item.highlight ? '#22D3EE' : '#94A3B8',
                fontWeight: isActive ? 600 : 500,
                fontSize: '13.5px',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={18} color={isActive ? '#38BDF8' : item.highlight ? '#22D3EE' : '#94A3B8'} />
              <span style={{ flex: 1 }}>{item.label}</span>
              {item.highlight && (
                <span style={{
                  fontSize: '9px',
                  fontWeight: 700,
                  backgroundColor: '#06B6D4',
                  color: '#000000',
                  padding: '2px 5px',
                  borderRadius: '4px',
                  textTransform: 'uppercase'
                }}>
                  Live
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* System Status Footer */}
      <div style={{ padding: '16px', borderTop: '1px solid #1E293B', backgroundColor: '#0B0F19' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10B981', boxShadow: '0 0 8px #10B981' }}></div>
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#E2E8F0' }}>Orchestrator Online</span>
        </div>
        <p style={{ fontSize: '11px', color: '#64748B' }}>10 Models Active • BigEarthNet Adapted</p>
      </div>
    </aside>
  );
};
