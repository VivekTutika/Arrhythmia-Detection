import { Link } from 'react-router-dom';
import { 
  Heart, 
  Brain, 
  Activity, 
  FileText, 
  Upload, 
  History,
  Settings,
  ArrowRight,
  CheckCircle,
  Zap,
  Shield,
  Clock,
  TrendingUp,
  ChevronRight,
  Sparkles,
  Cpu,
  AlertCircle
} from 'lucide-react';

const Home = () => {
  const features = [
    {
      icon: Brain,
      title: 'Deep Spiking Neural Network',
      description: 'Advanced DSNN architecture specifically designed for analyzing ECG signals with high accuracy.'
    },
    {
      icon: Zap,
      title: 'Real-time Analysis',
      description: 'Process ECG recordings quickly and get instant results with detailed diagnostics.'
    },
    {
      icon: Shield,
      title: 'Multi-class Detection',
      description: 'Identifies 6 different types of cardiac conditions including Normal Sinus Rhythm, Atrial Fibrillation, and more.'
    },
    {
      icon: Clock,
      title: 'Historical Tracking',
      description: 'Keep track of all your analyses with detailed history and trend monitoring.'
    }
  ];

  const arrhythmiaTypes = [
    { name: 'Normal Sinus Rhythm', color: '#10b981', description: 'Regular heart rhythm with normal rate' },
    { name: 'Atrial Fibrillation', color: '#f59e0b', description: 'Irregular and often rapid heart rhythm' },
    { name: 'Ventricular Arrhythmia', color: '#ef4444', description: 'Originating from the heart ventricles' },
    { name: 'Conduction Block', color: '#8b5cf6', description: 'Interrupted electrical signals in heart' },
    { name: 'Premature Contraction', color: '#ec4899', description: 'Early heartbeats causing irregularity' },
    { name: 'ST Segment Abnormality', color: '#6366f1', description: 'Indicates potential heart stress' }
  ];

  const processSteps = [
    {
      step: 1,
      title: 'Upload ECG',
      description: 'Upload your ECG file in EDF format or use sample data for testing.',
      icon: Upload
    },
    {
      step: 2,
      title: 'AI Analysis',
      description: 'Our DSNN model processes the signal and identifies cardiac patterns.',
      icon: Cpu
    },
    {
      step: 3,
      title: 'Get Results',
      description: 'View detailed results with confidence scores and recommendations.',
      icon: Activity
    }
  ];

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-badge">
            <Sparkles size={14} />
            <span>AI-Powered Healthcare</span>
          </div>
          <h1 className="hero-title">
            Early Detection of <span className="gradient-text">Arrhythmia</span> using Deep Spiking Neural Network
          </h1>
          <p className="hero-description">
            An advanced web-based application that analyzes Electrocardiogram (ECG) recordings 
            to detect various types of cardiac arrhythmias. Powered by cutting-edge Deep Spiking 
            Neural Network (DSNN) technology for accurate and reliable results.
          </p>
          <div className="hero-actions">
            <Link to="/upload" className="btn btn-primary btn-lg">
              <Upload size={20} />
              Start New Analysis
              <ArrowRight size={20} />
            </Link>
            <Link to="/dashboard" className="btn btn-secondary btn-lg">
              <Activity size={20} />
              View Dashboard
            </Link>
          </div>
          
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="stat-number">6+</span>
              <span className="stat-label">Cardiac Conditions</span>
            </div>
            <div className="hero-stat">
              <span className="stat-number">DSNN</span>
              <span className="stat-label">AI Model</span>
            </div>
            <div className="hero-stat">
              <span className="stat-number">24/7</span>
              <span className="stat-label">Availability</span>
            </div>
          </div>
        </div>
        
        <div className="hero-visual">
          <div className="ecg-animation">
            <svg viewBox="0 0 400 120" className="ecg-line">
              <path 
                d="M0,60 L50,60 L55,60 L60,30 L65,90 L70,40 L75,80 L80,60 L150,60 L155,60 L160,30 L165,90 L170,40 L175,80 L180,60 L250,60 L255,60 L260,30 L265,90 L270,40 L275,80 L280,60 L350,60 L355,60 L360,30 L365,90 L370,40 L375,80 L380,60 L400,60" 
                fill="none" 
                stroke="url(#ecgGradient)" 
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <defs>
                <linearGradient id="ecgGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#667eea" />
                  <stop offset="50%" stopColor="#764ba2" />
                  <stop offset="100%" stopColor="#667eea" />
                </linearGradient>
              </defs>
            </svg>
            <div className="ecg-dots">
              <span></span><span></span><span></span><span></span><span></span>
            </div>
          </div>
          <div className="hero-card">
            <div className="card-icon">
              <Heart size={24} />
            </div>
            <div className="card-content">
              <span className="card-label">Detection Status</span>
              <span className="card-value">Normal</span>
              <span className="card-confidence">98.5% Confidence</span>
            </div>
            <CheckCircle size={20} className="card-check" />
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="section">
        <div className="section-header">
          <h2>How It Works</h2>
          <p>Simple three-step process to get your cardiac analysis</p>
        </div>
        
        <div className="process-grid">
          {processSteps.map((item, index) => (
            <div key={index} className="process-card">
              <div className="process-icon">
                <item.icon size={28} />
              </div>
              <div className="process-step">Step {item.step}</div>
              <h3>{item.title}</h3>
              <p>{item.description}</p>
              {index < processSteps.length - 1 && (
                <ChevronRight className="process-arrow" size={20} />
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="section">
        <div className="section-header">
          <h2>Key Features</h2>
          <p>Advanced capabilities for accurate cardiac analysis</p>
        </div>
        
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className="feature-icon">
                <feature.icon size={24} />
              </div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Detectable Conditions Section */}
      <section className="section">
        <div className="section-header">
          <h2>Detectable Conditions</h2>
          <p>Six types of cardiac arrhythmias our DSNN model can detect</p>
        </div>
        
        <div className="conditions-grid">
          {arrhythmiaTypes.map((condition, index) => (
            <div key={index} className="condition-card">
              <div 
                className="condition-indicator" 
                style={{ backgroundColor: condition.color }}
              />
              <div className="condition-content">
                <h4>{condition.name}</h4>
                <p>{condition.description}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Technical Info Section */}
      <section className="section">
        <div className="section-header">
          <h2>About the Technology</h2>
          <p>Understanding the DSNN architecture</p>
        </div>
        
        <div className="tech-grid">
          <div className="tech-card">
            <div className="tech-icon">
              <Brain size={32} />
            </div>
            <h3>Deep Spiking Neural Network</h3>
            <p>
              DSNN combines the benefits of deep learning with spiking neural networks, 
              mimicking the way biological neurons communicate. This results in more 
              efficient processing of temporal data like ECG signals.
            </p>
            <ul className="tech-features">
              <li><CheckCircle size={14} /> Efficient temporal processing</li>
              <li><CheckCircle size={14} /> Low power consumption</li>
              <li><CheckCircle size={14} /> High accuracy in pattern recognition</li>
              <li><CheckCircle size={14} /> Robust to signal noise</li>
            </ul>
          </div>
          
          <div className="tech-card">
            <div className="tech-icon">
              <TrendingUp size={32} />
            </div>
            <h3>Working Mechanism</h3>
            <p>
              The system processes ECG data through multiple convolutional layers that 
              extract features from the signal, followed by fully connected layers that 
              classify the patterns into different cardiac conditions.
            </p>
            <ul className="tech-features">
              <li><CheckCircle size={14} /> Multi-layer feature extraction</li>
              <li><CheckCircle size={14} /> Sequence shifting for data augmentation</li>
              <li><CheckCircle size={14} /> Voting mechanism for robust predictions</li>
              <li><CheckCircle size={14} /> Confidence scoring for each prediction</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Quick Links Section */}
      <section className="section quick-links-section">
        <div className="section-header">
          <h2>Quick Links</h2>
          <p>Navigate to different parts of the application</p>
        </div>
        
        <div className="quick-links-grid">
          <Link to="/upload" className="quick-link-card">
            <Upload size={24} />
            <span>Upload ECG</span>
            <ArrowRight size={16} className="link-arrow" />
          </Link>
          <Link to="/history" className="quick-link-card">
            <History size={24} />
            <span>View History</span>
            <ArrowRight size={16} className="link-arrow" />
          </Link>
          <Link to="/dashboard" className="quick-link-card">
            <Activity size={24} />
            <span>Dashboard</span>
            <ArrowRight size={16} className="link-arrow" />
          </Link>
          <Link to="/settings" className="quick-link-card">
            <Settings size={24} />
            <span>Settings</span>
            <ArrowRight size={16} className="link-arrow" />
          </Link>
        </div>
      </section>

      {/* Disclaimer Banner */}
      <section className="disclaimer-banner">
        <AlertCircle size={20} />
        <div>
          <strong>Important Disclaimer</strong>
          <p>
            This application is for research and educational purposes only. 
            The results should not be used as the sole basis for medical decisions. 
            Always consult with a qualified healthcare professional for proper diagnosis 
            and treatment.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="home-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <Heart size={24} className="footer-icon" />
            <div>
              <span className="footer-title">Arrhythmia Detector</span>
              <span className="footer-subtitle">DSNN v1.0.0</span>
            </div>
          </div>
          <p className="footer-text">
            Early Detection of Arrhythmia using Deep Spiking Neural Network
          </p>
        </div>
      </footer>

      <style>{`
        .home-page {
          padding: 0;
          max-width: 1400px;
          margin: 0 auto;
        }

        /* Hero Section */
        .hero-section {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 48px;
          padding: 48px 0;
          align-items: center;
        }

        @media (max-width: 1024px) {
          .hero-section {
            grid-template-columns: 1fr;
            text-align: center;
          }
        }

        .hero-badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 8px 16px;
          background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
          border: 1px solid rgba(102, 126, 234, 0.2);
          border-radius: 50px;
          font-size: 14px;
          color: var(--primary-color);
          margin-bottom: 24px;
        }

        .hero-title {
          font-size: 48px;
          font-weight: 700;
          line-height: 1.2;
          margin-bottom: 24px;
          color: var(--text-primary);
        }

        @media (max-width: 768px) {
          .hero-title {
            font-size: 32px;
          }
        }

        .gradient-text {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .hero-description {
          font-size: 18px;
          line-height: 1.8;
          color: var(--text-secondary);
          margin-bottom: 32px;
        }

        .hero-actions {
          display: flex;
          gap: 16px;
          margin-bottom: 48px;
        }

        @media (max-width: 1024px) {
          .hero-actions {
            justify-content: center;
          }
        }

        .btn-lg {
          padding: 14px 28px;
          font-size: 16px;
        }

        .hero-stats {
          display: flex;
          gap: 48px;
        }

        @media (max-width: 1024px) {
          .hero-stats {
            justify-content: center;
          }
        }

        .hero-stat {
          display: flex;
          flex-direction: column;
        }

        .stat-number {
          font-size: 32px;
          font-weight: 700;
          color: var(--primary-color);
        }

        .stat-label {
          font-size: 14px;
          color: var(--text-secondary);
        }

        .hero-visual {
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 24px;
        }

        .ecg-animation {
          width: 100%;
          max-width: 400px;
          background: var(--background-card);
          border-radius: 16px;
          padding: 24px;
          border: 1px solid var(--border-color);
        }

        .ecg-line {
          width: 100%;
          height: auto;
          animation: drawECG 2s ease-in-out infinite;
        }

        @keyframes drawECG {
          0%, 100% {
            opacity: 0.5;
          }
          50% {
            opacity: 1;
          }
        }

        .hero-card {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 20px 24px;
          background: var(--background-card);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .hero-card .card-icon {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          background: rgba(16, 185, 129, 0.1);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--success-color);
        }

        .hero-card .card-content {
          display: flex;
          flex-direction: column;
        }

        .hero-card .card-label {
          font-size: 12px;
          color: var(--text-secondary);
        }

        .hero-card .card-value {
          font-size: 18px;
          font-weight: 600;
          color: var(--success-color);
        }

        .hero-card .card-confidence {
          font-size: 12px;
          color: var(--text-secondary);
        }

        .hero-card .card-check {
          color: var(--success-color);
          margin-left: auto;
        }

        /* Section Styles */
        .section {
          padding: 64px 0;
        }

        .section-header {
          text-align: center;
          margin-bottom: 48px;
        }

        .section-header h2 {
          font-size: 36px;
          font-weight: 700;
          margin-bottom: 12px;
          color: var(--text-primary);
        }

        .section-header p {
          font-size: 18px;
          color: var(--text-secondary);
        }

        /* Process Grid */
        .process-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 24px;
          position: relative;
        }

        @media (max-width: 768px) {
          .process-grid {
            grid-template-columns: 1fr;
          }
        }

        .process-card {
          background: var(--background-card);
          border: 1px solid var(--border-color);
          border-radius: 16px;
          padding: 32px;
          text-align: center;
          position: relative;
        }

        .process-icon {
          width: 64px;
          height: 64px;
          border-radius: 16px;
          background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 20px;
          color: var(--primary-color);
        }

        .process-step {
          font-size: 12px;
          font-weight: 600;
          color: var(--primary-color);
          text-transform: uppercase;
          letter-spacing: 1px;
          margin-bottom: 8px;
        }

        .process-card h3 {
          font-size: 20px;
          font-weight: 600;
          margin-bottom: 12px;
          color: var(--text-primary);
        }

        .process-card p {
          font-size: 14px;
          color: var(--text-secondary);
          line-height: 1.6;
        }

        .process-arrow {
          display: none;
          position: absolute;
          right: -20px;
          top: 50%;
          transform: translateY(-50%);
          color: var(--primary-color);
          z-index: 1;
        }

        @media (min-width: 769px) and (max-width: 1024px) {
          .process-arrow {
            display: block;
          }
        }

        /* Features Grid */
        .features-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 24px;
        }

        @media (max-width: 1024px) {
          .features-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 640px) {
          .features-grid {
            grid-template-columns: 1fr;
          }
        }

        .feature-card {
          background: var(--background-card);
          border: 1px solid var(--border-color);
          border-radius: 16px;
          padding: 24px;
          transition: all 0.3s ease;
        }

        .feature-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        }

        .feature-icon {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 16px;
          color: var(--primary-color);
        }

        .feature-card h3 {
          font-size: 16px;
          font-weight: 600;
          margin-bottom: 8px;
          color: var(--text-primary);
        }

        .feature-card p {
          font-size: 14px;
          color: var(--text-secondary);
          line-height: 1.6;
        }

        /* Conditions Grid */
        .conditions-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
        }

        @media (max-width: 1024px) {
          .conditions-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 640px) {
          .conditions-grid {
            grid-template-columns: 1fr;
          }
        }

        .condition-card {
          display: flex;
          align-items: flex-start;
          gap: 16px;
          padding: 20px;
          background: var(--background-card);
          border: 1px solid var(--border-color);
          border-radius: 12px;
        }

        .condition-indicator {
          width: 4px;
          height: 100%;
          min-height: 48px;
          border-radius: 2px;
          flex-shrink: 0;
        }

        .condition-content h4 {
          font-size: 15px;
          font-weight: 600;
          margin-bottom: 4px;
          color: var(--text-primary);
        }

        .condition-content p {
          font-size: 13px;
          color: var(--text-secondary);
          line-height: 1.5;
        }

        /* Tech Grid */
        .tech-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 24px;
        }

        @media (max-width: 768px) {
          .tech-grid {
            grid-template-columns: 1fr;
          }
        }

        .tech-card {
          background: var(--background-card);
          border: 1px solid var(--border-color);
          border-radius: 16px;
          padding: 32px;
        }

        .tech-icon {
          width: 64px;
          height: 64px;
          border-radius: 16px;
          background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 20px;
          color: var(--primary-color);
        }

        .tech-card h3 {
          font-size: 20px;
          font-weight: 600;
          margin-bottom: 12px;
          color: var(--text-primary);
        }

        .tech-card p {
          font-size: 14px;
          color: var(--text-secondary);
          line-height: 1.7;
          margin-bottom: 20px;
        }

        .tech-features {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .tech-features li {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: var(--text-secondary);
          margin-bottom: 8px;
        }

        .tech-features li svg {
          color: var(--success-color);
          flex-shrink: 0;
        }

        /* Quick Links */
        .quick-links-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
        }

        @media (max-width: 768px) {
          .quick-links-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        .quick-link-card {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
          padding: 24px;
          background: var(--background-card);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          text-decoration: none;
          color: var(--text-primary);
          transition: all 0.3s ease;
        }

        .quick-link-card:hover {
          background: var(--background-secondary);
          transform: translateY(-2px);
        }

        .quick-link-card svg {
          color: var(--primary-color);
        }

        .quick-link-card span {
          font-size: 14px;
          font-weight: 500;
        }

        .quick-link-card .link-arrow {
          opacity: 0;
          transform: translateX(-4px);
          transition: all 0.3s ease;
          color: var(--text-secondary);
        }

        .quick-link-card:hover .link-arrow {
          opacity: 1;
          transform: translateX(0);
        }

        /* Disclaimer Banner */
        .disclaimer-banner {
          display: flex;
          align-items: flex-start;
          gap: 16px;
          padding: 20px 24px;
          background: rgba(245, 158, 11, 0.1);
          border: 1px solid rgba(245, 158, 11, 0.3);
          border-radius: 12px;
          margin: 48px 0;
        }

        .disclaimer-banner svg {
          color: #f59e0b;
          flex-shrink: 0;
          margin-top: 2px;
        }

        .disclaimer-banner strong {
          display: block;
          font-size: 14px;
          color: #f59e0b;
          margin-bottom: 4px;
        }

        .disclaimer-banner p {
          font-size: 13px;
          color: var(--text-secondary);
          margin: 0;
          line-height: 1.6;
        }

        /* Footer */
        .home-footer {
          border-top: 1px solid var(--border-color);
          padding: 32px 0;
          margin-top: 48px;
        }

        .footer-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 16px;
        }

        .footer-brand {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .footer-icon {
          color: var(--primary-color);
        }

        .footer-title {
          display: block;
          font-size: 16px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .footer-subtitle {
          display: block;
          font-size: 12px;
          color: var(--text-secondary);
        }

        .footer-text {
          font-size: 14px;
          color: var(--text-secondary);
        }
      `}</style>
    </div>
  );
};

export default Home;
