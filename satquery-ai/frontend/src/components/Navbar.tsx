import React from 'react';
import { Activity, ShieldCheck, Database, Terminal } from 'lucide-react';

interface NavbarProps {
  title: string;
  subtitle: string;
}

export const Navbar: React.FC<NavbarProps> = ({ title, subtitle }) => {
  return (
    <header style={{
      height: '64px',
      borderBottom: '1px solid #1E293B',
      backgroundColor: 'rgba(15, 23, 42, 0.8)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 28px',
      position: 'sticky',
      top: 0,
      zIndex: 10
    }}>
      <div>
        <h2 style={{ fontSize: '17px', fontWeight: 700, color: '#F8FAFC' }}>{title}</h2>
        <p style={{ fontSize: '11.5px', color: '#64748B' }}>{subtitle}</p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 12px',
          backgroundColor: '#0B0F19',
          border: '1px solid #1E293B',
          borderRadius: '6px',
          fontSize: '12px',
          color: '#94A3B8'
        }}>
          <Database size={14} color="#38BDF8" />
          <span>BigEarthNet-v1.0 (12-Band)</span>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 12px',
          backgroundColor: '#0B0F19',
          border: '1px solid #1E293B',
          borderRadius: '6px',
          fontSize: '12px',
          color: '#94A3B8'
        }}>
          <ShieldCheck size={14} color="#10B981" />
          <span>Zero Fake AI Verified</span>
        </div>
      </div>
    </header>
  );
};
