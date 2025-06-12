from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import os
import re
import math
import numpy as np
from urllib.parse import urlparse
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import hashlib
import socket
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
CONFIG = {
    'max_dataset_rows': 3000,
    'test_size': 0.2,
    'random_state': 42,
    'model_confidence_threshold': 0.6,
    'max_features': 3000
}

class EnhancedPhishingDetector:
    def __init__(self):
        # Balanced weights that sum to 100
        self.weights = {
            'domain_reputation': 25,
            'url_entropy': 18,
            'suspicious_patterns': 20,
            'domain_features': 15,
            'security_indicators': 12,
            'lexical_analysis': 10
        }
        
        # Fixed threshold logic
        self.thresholds = {
            'safe': 20,        # 0-20: Safe
            'suspicious': 45,  # 21-45: Suspicious
            'phishing': 46     # 46-100: Phishing
        }
        
        # Enhanced keyword lists with weights
        self.phishing_keywords = {
            'financial': {
                'keywords': ['bank', 'paypal', 'payment', 'billing', 'invoice', 'refund', 'credit', 'debit', 'transfer', 'account'],
                'weight': 3.0
            },
            'security': {
                'keywords': ['security', 'verify', 'confirm', 'authenticate', 'validate', 'suspended', 'locked', 'blocked', 'unauthorized'],
                'weight': 2.5
            },
            'urgency': {
                'keywords': ['urgent', 'immediate', 'expire', 'limited', 'action', 'deadline', 'warning', 'alert', 'now'],
                'weight': 2.0
            },
            'brands': {
                'keywords': ['amazon', 'microsoft', 'google', 'apple', 'facebook', 'netflix', 'ebay', 'instagram', 'whatsapp'],
                'weight': 1.8
            },
            'actions': {
                'keywords': ['login', 'signin', 'update', 'renewal', 'activation', 'unlock', 'restore', 'download', 'install'],
                'weight': 1.5
            }
        }
        
        self.legitimate_domains = {
            'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com',
            'paypal.com', 'ebay.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'netflix.com', 'spotify.com', 'dropbox.com', 'github.com', 'reddit.com',
            'wikipedia.org', 'stackoverflow.com', 'youtube.com', 'gmail.com', 'yahoo.com',
            'bank.com', 'chase.com', 'wellsfargo.com', 'bankofamerica.com'
        }
        
        # Suspicious TLDs with risk scores
        self.suspicious_tlds = {
            'tk': 0.9, 'ml': 0.9, 'ga': 0.9, 'cf': 0.9,
            'xyz': 0.7, 'top': 0.6, 'click': 0.8, 'download': 0.8,
            'work': 0.5, 'bid': 0.7, 'win': 0.6, 'date': 0.7
        }
        
        self.load_datasets()
    
    def load_datasets(self):
        """Load whitelist and phishing datasets with enhanced error handling"""
        self.whitelist_domains = set()
        try:
            if os.path.exists('whitelist_domains.csv'):
                whitelist_df = pd.read_csv('whitelist_domains.csv')
                if 'domain' in whitelist_df.columns:
                    self.whitelist_domains = set(whitelist_df['domain'].dropna().str.lower())
                    print(f"✅ Loaded {len(self.whitelist_domains)} whitelist domains")
        except Exception as e:
            print(f"⚠️ Could not load whitelist: {e}")
        
        self.phishing_domains = set()
        try:
            if os.path.exists('PhiUSIIL_Phishing_URL_Dataset.csv'):
                df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', nrows=CONFIG['max_dataset_rows'])
                if 'Domain' in df.columns and 'label' in df.columns:
                    phishing_data = df[df['label'] == 1]['Domain'].dropna()
                    self.phishing_domains = set(phishing_data.str.lower())
                    print(f"✅ Loaded {len(self.phishing_domains)} known phishing domains")
        except Exception as e:
            print(f"⚠️ Could not load phishing dataset: {e}")
        
        # Default phishing patterns if no dataset available
        if not self.phishing_domains:
            self.phishing_domains = {
                'phishing-example.tk', 'fake-paypal.com', 'microsoft-support.xyz',
                'amazon-security.ml', 'google-verify.ga', 'secure-bank.tk'
            }
    
    def calculate_url_entropy(self, url):
        """Calculate Shannon entropy with improved normalization"""
        try:
            # Clean URL for entropy calculation
            clean_url = re.sub(r'https?://', '', url.lower())
            clean_url = re.sub(r'[./\-_]', '', clean_url)
            
            if len(clean_url) == 0:
                return 0.0
            
            # Calculate character frequency
            char_counts = {}
            for char in clean_url:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Calculate Shannon entropy
            entropy = 0.0
            length = len(clean_url)
            for count in char_counts.values():
                probability = count / length
                if probability > 0:  # Avoid log(0)
                    entropy -= probability * math.log(probability, 2)
            
            return round(entropy, 3)
        except Exception as e:
            print(f"Entropy calculation error: {e}")
            return 0.0
    
    def extract_domain_features(self, url):
        """Extract comprehensive domain features with validation"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Handle port in domain
            if ':' in domain and not domain.startswith('['):
                domain = domain.split(':')[0]
            
            extracted = tldextract.extract(url)
            
            features = {
                'domain': domain,
                'subdomain': extracted.subdomain or '',
                'domain_name': extracted.domain or '',
                'suffix': extracted.suffix or '',
                'subdomain_count': len([s for s in extracted.subdomain.split('.') if s]) if extracted.subdomain else 0,
                'domain_length': len(extracted.domain) if extracted.domain else 0,
                'has_ip': self.is_ip_address(domain),
                'has_port': ':' in parsed.netloc and not parsed.netloc.startswith('['),
                'url_length': len(url),
                'path_length': len(parsed.path),
                'query_length': len(parsed.query) if parsed.query else 0,
                'fragment_length': len(parsed.fragment) if parsed.fragment else 0,
                'has_suspicious_tld': extracted.suffix.lower() in self.suspicious_tlds if extracted.suffix else False
            }
            
            return features
        except Exception as e:
            print(f"Domain feature extraction error: {e}")
            return {}
    
    def is_ip_address(self, domain):
        """Enhanced IP address detection"""
        # IPv4 pattern
        ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        # IPv6 pattern (simplified)
        ipv6_pattern = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        
        return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))
    
    def check_suspicious_patterns(self, url):
        """Enhanced pattern detection with weighted scoring"""
        patterns = {
            'url_shortener': {
                'pattern': r'(bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|short\.link|tiny\.cc)',
                'weight': 2
            },
            'suspicious_tld': {
                'pattern': r'\.(tk|ml|ga|cf|xyz|top|click|download|work|bid|win|date)$',
                'weight': 3
            },
            'homograph_attack': {
                'pattern': r'[а-я]',  # Cyrillic characters
                'weight': 4
            },
            'excessive_hyphens': {
                'pattern': r'-{2,}',
                'weight': 2
            },
            'numeric_domain': {
                'pattern': r'[0-9]{4,}',
                'weight': 2
            },
            'suspicious_ports': {
                'pattern': r':(8080|8443|3128|1080|8000|9999)',
                'weight': 3
            },
            'data_uri': {
                'pattern': r'^data:',
                'weight': 4
            },
            'javascript_uri': {
                'pattern': r'^javascript:',
                'weight': 4
            },
            'multiple_subdomains': {
                'custom': lambda u: len(urlparse(u).netloc.split('.')) > 5,
                'weight': 2
            }
        }
        
        detected = {}
        total_score = 0
        
        for pattern_name, config in patterns.items():
            if 'custom' in config:
                # Custom function check
                try:
                    if config['custom'](url):
                        detected[pattern_name] = True
                        total_score += config['weight']
                except:
                    pass
            elif 'pattern' in config:
                # Regex pattern check
                if re.search(config['pattern'], url, re.IGNORECASE):
                    detected[pattern_name] = True
                    total_score += config['weight']
        
        # Normalize score to weight limit
        max_score = min(total_score, self.weights['suspicious_patterns'])
        
        return detected, max_score
    
    def analyze_keyword_density(self, url):
        """Enhanced keyword analysis with weighted categories"""
        url_lower = url.lower()
        detected_keywords = {}
        total_score = 0
        
        for category, config in self.phishing_keywords.items():
            keywords = config['keywords']
            weight = config['weight']
            
            found = [kw for kw in keywords if kw in url_lower]
            if found:
                detected_keywords[category] = found
                # Score based on number of keywords and category weight
                category_score = len(found) * weight
                total_score += category_score
        
        # Normalize to maximum weight
        normalized_score = min(total_score, self.weights['lexical_analysis'])
        
        return detected_keywords, normalized_score
    
    def check_domain_reputation(self, domain):
        """Enhanced domain reputation with detailed scoring"""
        score = 0
        details = []
        confidence = 0
        
        # Remove port if present
        clean_domain = domain.split(':')[0] if ':' in domain else domain
        
        # Check against known phishing domains (highest priority)
        if clean_domain in self.phishing_domains:
            score = self.weights['domain_reputation']
            details.append("⚠️ Domain found in phishing database")
            confidence = 95
        
        # Check against whitelist (trusted domains)
        elif clean_domain in self.whitelist_domains:
            score = 0
            details.append("✅ Domain in trusted whitelist")
            confidence = 90
        
        # Check against well-known legitimate domains
        elif clean_domain in self.legitimate_domains:
            score = 0
            details.append("✅ Well-known legitimate domain")
            confidence = 85
        
        else:
            # Check for typosquatting
            typo_score, typo_detail, typo_confidence = self.check_typosquatting(clean_domain)
            score += typo_score
            details.append(typo_detail)
            confidence = typo_confidence
        
        return score, '; '.join(details), confidence
    
    def check_typosquatting(self, domain):
        """Improved typosquatting detection with confidence scoring"""
        try:
            extracted = tldextract.extract(domain)
            domain_name = extracted.domain.lower()
            
            if not domain_name:
                return 0, "No domain name to analyze", 50
            
            best_similarity = 0
            best_match = ""
            
            for legitimate in self.legitimate_domains:
                legit_extracted = tldextract.extract(legitimate)
                legit_name = legit_extracted.domain.lower()
                
                if domain_name == legit_name:
                    continue
                
                # Calculate similarity
                distance = self.levenshtein_distance(domain_name, legit_name)
                max_len = max(len(domain_name), len(legit_name))
                
                if max_len == 0:
                    continue
                    
                similarity = 1 - (distance / max_len)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = legitimate
            
            # Enhanced similarity thresholds
            if best_similarity >= 0.85:
                return 18, f"🚨 High typosquatting risk of {best_match} (similarity: {best_similarity:.2f})", 90
            elif best_similarity >= 0.75:
                return 12, f"⚠️ Possible typosquatting of {best_match} (similarity: {best_similarity:.2f})", 70
            elif best_similarity >= 0.65:
                return 6, f"❓ Minor similarity to {best_match} (similarity: {best_similarity:.2f})", 50
            else:
                return 0, "✅ No typosquatting patterns detected", 20
                
        except Exception as e:
            return 0, f"Could not analyze typosquatting: {e}", 30
    
    def levenshtein_distance(self, s1, s2):
        """Optimized Levenshtein distance calculation"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def analyze_security_indicators(self, url):
        """Enhanced security analysis"""
        parsed = urlparse(url)
        score = 0
        details = []
        
        # Protocol security analysis
        if parsed.scheme == 'http':
            score += 8
            details.append("🔓 Insecure HTTP protocol")
        elif parsed.scheme == 'https':
            details.append("🔒 Secure HTTPS protocol")
        elif parsed.scheme in ['ftp', 'file']:
            score += 6
            details.append(f"⚠️ Unusual protocol: {parsed.scheme}")
        elif parsed.scheme:
            score += 4
            details.append(f"❓ Unknown protocol: {parsed.scheme}")
        
        # Port analysis
        if ':' in parsed.netloc and not parsed.netloc.startswith('['):
            port_match = re.search(r':(\d+)', parsed.netloc)
            if port_match:
                port = int(port_match.group(1))
                if port in [8080, 8443, 3128, 1080, 8000, 9999]:
                    score += 4
                    details.append(f"⚠️ Suspicious port: {port}")
                elif port not in [80, 443]:
                    score += 2
                    details.append(f"❓ Unusual port: {port}")
        
        normalized_score = min(score, self.weights['security_indicators'])
        
        return normalized_score, '; '.join(details) if details else "✅ Standard security indicators"
    
    def calculate_confidence(self, total_score, classification):
        """Improved confidence calculation with realistic ranges"""
        max_score = sum(self.weights.values())
        score_ratio = total_score / max_score
        
        if classification == 'safe':
            # Higher confidence for lower scores
            if total_score <= 5:
                confidence = 95
            elif total_score <= 10:
                confidence = 85
            elif total_score <= self.thresholds['safe']:
                confidence = 75 - (total_score - 10) * 2
            else:
                confidence = 60
                
        elif classification == 'suspicious':
            # Moderate confidence for middle range
            mid_point = (self.thresholds['safe'] + self.thresholds['suspicious']) / 2
            if total_score <= mid_point:
                confidence = 60 + (total_score - self.thresholds['safe']) * 0.8
            else:
                confidence = 70 + (total_score - mid_point) * 0.6
                
        else:  # phishing
            # High confidence for high scores
            if total_score >= 80:
                confidence = 95
            elif total_score >= 60:
                confidence = 85 + (total_score - 60) * 0.5
            else:
                confidence = 75 + (total_score - self.thresholds['suspicious']) * 0.7
        
        return max(50, min(95, round(confidence, 1)))
    
    def calculate_risk_score(self, url):
        """Main analysis function with improved error handling and scoring"""
        if not url or not isinstance(url, str):
            return None
        
        try:
            # Extract domain features
            domain_features = self.extract_domain_features(url)
            if not domain_features or not domain_features.get('domain'):
                return None
            
            analysis = {
                'url': url,
                'domain': domain_features['domain'],
                'timestamp': datetime.now().isoformat(),
                'methods': {},
                'total_score': 0,
                'max_possible_score': sum(self.weights.values()),
                'classification': 'safe',
                'confidence': 0,
                'risk_level': 'Low',
                'domain_features': domain_features
            }
            
            # 1. Domain reputation check (highest weight)
            rep_score, rep_details, rep_confidence = self.check_domain_reputation(domain_features['domain'])
            analysis['methods']['domain_reputation'] = {
                'score': rep_score,
                'max_score': self.weights['domain_reputation'],
                'details': rep_details,
                'detected': rep_score > 0,
                'confidence': rep_confidence,
                'weight': 'High'
            }
            
            # 2. URL entropy analysis
            entropy = self.calculate_url_entropy(url)
            entropy_score = 0
            entropy_details = f"URL entropy: {entropy}"
            
            if entropy > 4.5:
                entropy_score = self.weights['url_entropy']
                entropy_details += " (Very High - Highly obfuscated)"
            elif entropy > 3.8:
                entropy_score = int(self.weights['url_entropy'] * 0.8)
                entropy_details += " (High - Likely obfuscated)"
            elif entropy > 3.0:
                entropy_score = int(self.weights['url_entropy'] * 0.5)
                entropy_details += " (Medium - Some obfuscation)"
            elif entropy > 2.5:
                entropy_score = int(self.weights['url_entropy'] * 0.3)
                entropy_details += " (Low-Medium - Minor obfuscation)"
            else:
                entropy_details += " (Normal - No obfuscation detected)"
            
            analysis['methods']['url_entropy'] = {
                'score': entropy_score,
                'max_score': self.weights['url_entropy'],
                'details': entropy_details,
                'detected': entropy_score > 0,
                'entropy_value': entropy,
                'weight': 'High'
            }
            
            # 3. Suspicious patterns
            patterns, pattern_score = self.check_suspicious_patterns(url)
            analysis['methods']['suspicious_patterns'] = {
                'score': pattern_score,
                'max_score': self.weights['suspicious_patterns'],
                'details': f"Detected: {list(patterns.keys())}" if patterns else "No suspicious patterns detected",
                'detected': pattern_score > 0,
                'patterns': patterns,
                'weight': 'High'
            }
            
            # 4. Domain features analysis
            domain_score = 0
            domain_details = []
            
            if domain_features['has_ip']:
                domain_score += 10
                domain_details.append("🚨 Using IP address instead of domain")
            
            if domain_features['subdomain_count'] > 3:
                points = min(5, domain_features['subdomain_count'] - 3)
                domain_score += points
                domain_details.append(f"⚠️ Many subdomains: {domain_features['subdomain_count']}")
            
            if domain_features['url_length'] > 150:
                domain_score += 5
                domain_details.append(f"🔗 Very long URL: {domain_features['url_length']} chars")
            elif domain_features['url_length'] > 100:
                domain_score += 3
                domain_details.append(f"🔗 Long URL: {domain_features['url_length']} chars")
            
            if domain_features['has_suspicious_tld']:
                tld_risk = self.suspicious_tlds.get(domain_features['suffix'].lower(), 0.5)
                domain_score += int(5 * tld_risk)
                domain_details.append(f"⚠️ Suspicious TLD: .{domain_features['suffix']}")
            
            domain_score = min(domain_score, self.weights['domain_features'])
            
            analysis['methods']['domain_features'] = {
                'score': domain_score,
                'max_score': self.weights['domain_features'],
                'details': '; '.join(domain_details) if domain_details else "✅ Normal domain characteristics",
                'detected': domain_score > 0,
                'weight': 'Medium'
            }
            
            # 5. Security indicators
            security_score, security_details = self.analyze_security_indicators(url)
            analysis['methods']['security_indicators'] = {
                'score': security_score,
                'max_score': self.weights['security_indicators'],
                'details': security_details,
                'detected': security_score > 0,
                'weight': 'Medium'
            }
            
            # 6. Lexical analysis
            keywords, keyword_score = self.analyze_keyword_density(url)
            keyword_count = sum(len(v) for v in keywords.values()) if keywords else 0
            
            analysis['methods']['lexical_analysis'] = {
                'score': keyword_score,
                'max_score': self.weights['lexical_analysis'],
                'details': f"Found {keyword_count} suspicious keywords in {len(keywords)} categories" if keywords else "No suspicious keywords detected",
                'detected': keyword_score > 0,
                'keywords': keywords,
                'weight': 'Low'
            }
            
            # Calculate total score and classification
            analysis['total_score'] = sum(method['score'] for method in analysis['methods'].values())
            
            # Fixed classification logic
            if analysis['total_score'] <= self.thresholds['safe']:
                analysis['classification'] = 'safe'
                analysis['risk_level'] = 'Low'
            elif analysis['total_score'] <= self.thresholds['suspicious']:
                analysis['classification'] = 'suspicious'
                analysis['risk_level'] = 'Medium'
            else:  # > 45
                analysis['classification'] = 'phishing'
                analysis['risk_level'] = 'High'
            
            # Calculate improved confidence
            analysis['confidence'] = self.calculate_confidence(analysis['total_score'], analysis['classification'])
            analysis['score_percentage'] = round((analysis['total_score'] / analysis['max_possible_score']) * 100, 1)
            
            return analysis
            
        except Exception as e:
            print(f"Error in risk score calculation: {e}")
            import traceback
            traceback.print_exc()
            return None

# Initialize detector
detector = EnhancedPhishingDetector()

def load_ml_models():
    """Load and train ML models with improved error handling"""
    try:
        if not os.path.exists('PhiUSIIL_Phishing_URL_Dataset.csv'):
            print("⚠️ Dataset file not found. ML models will not be available.")
            return None, None, None, 0, 0
            
        print("📊 Loading dataset...")
        df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', nrows=CONFIG['max_dataset_rows'])
        
        if 'URL' not in df.columns or 'label' not in df.columns:
            print("❌ Required columns (URL, label) not found in dataset.")
            return None, None, None, 0, 0
        
        # Clean and prepare data
        df = df.dropna(subset=['URL', 'label'])
        df = df[df['label'].isin([0, 1])]  # Only binary classification
        
        urls = df['URL'].values
        labels = df['label'].values
        
        print(f"🎯 Training on {len(urls)} samples...")
        print(f"   - Safe URLs: {sum(labels == 0)}")
        print(f"   - Phishing URLs: {sum(labels == 1)}")
        
        # Feature extraction
        vectorizer = TfidfVectorizer(
            analyzer='char', 
            ngram_range=(2, 4),
            max_features=CONFIG['max_features'],
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        
        X = vectorizer.fit_transform(urls)
        print(f"📈 Feature matrix shape: {X.shape}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, 
            test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state'],
            stratify=labels
        )
        
        # Train SVM
        print("🤖 Training SVM model...")
        svm_model = SVC(
            probability=True, 
            kernel='linear',
            C=0.1,
            random_state=CONFIG['random_state'],
            max_iter=1000
        )
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        
        # Train Decision Tree
        print("🌳 Training Decision Tree model...")
        dt_model = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=CONFIG['random_state']
        )
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        
        print(f"✅ Model Training Complete:")
        print(f"   - SVM Accuracy: {svm_accuracy:.3f}")
        print(f"   - Decision Tree Accuracy: {dt_accuracy:.3f}")
        
        return vectorizer, svm_model, dt_model, svm_accuracy, dt_accuracy
        
    except Exception as e:
        print(f"❌ Error loading ML models: {e}")
        return None, None, None, 0, 0

# Load models
print("🚀 Initializing ML models...")
vectorizer, svm_model, dt_model, svm_accuracy, dt_accuracy = load_ml_models()

def predict_with_ml(url, model, model_name):
    """Enhanced ML prediction with better error handling"""
    if model is None or vectorizer is None:
        return 'unknown', 0.0, f"{model_name} model not available"
    
    try:
        X = vectorizer.transform([url])
        proba = model.predict_proba(X)[0]
        
        # Handle different probability array structures
        if len(proba) == 2:
            safe_prob = proba[0]
            phishing_prob = proba[1]
        else:
            phishing_prob = proba[1] if len(proba) > 1 else 0
            safe_prob = 1 - phishing_prob
        
        # Determine classification with improved logic
        if phishing_prob >= CONFIG['model_confidence_threshold']:
            classification = 'phishing'
            confidence = phishing_prob * 100
        elif phishing_prob >= 0.3:
            classification = 'suspicious'
            confidence = phishing_prob * 100
        else:
            classification = 'safe'
            confidence = safe_prob * 100
        
        return classification, round(confidence, 1), f"{model_name}: {confidence:.1f}% confidence"
        
    except Exception as e:
        return 'error', 0.0, f"{model_name} prediction failed: {str(e)}"

def calculate_hybrid_score(heuristic_result, svm_result, dt_result):
    """Fixed hybrid score calculation with proper weighting and validation"""
    if not heuristic_result:
        return None
    
    # Base weights for different methods
    weights = {
        'heuristic': 0.5,  # 50% weight for heuristic analysis
        'svm': 0.3,        # 30% weight for SVM
        'dt': 0.2          # 20% weight for Decision Tree
    }
    
    # Convert classifications to numeric scores (0-100)
    def classification_to_score(classification):
        if classification == 'safe':
            return 0
        elif classification == 'suspicious':
            return 50
        elif classification == 'phishing':
            return 100
        else:
            return 25  # unknown/error cases
    
    # Calculate weighted score
    heuristic_score = heuristic_result['total_score']
    total_weighted_score = heuristic_score * weights['heuristic']
    
    # Add ML predictions if available
    if svm_result[0] != 'unknown' and svm_result[0] != 'error':
        svm_score = classification_to_score(svm_result[0])
        total_weighted_score += svm_score * weights['svm']
    else:
        # Redistribute SVM weight to heuristic if SVM unavailable
        total_weighted_score += heuristic_score * weights['svm']
    
    if dt_result[0] != 'unknown' and dt_result[0] != 'error':
        dt_score = classification_to_score(dt_result[0])
        total_weighted_score += dt_score * weights['dt']
    else:
        # Redistribute DT weight to heuristic if DT unavailable
        total_weighted_score += heuristic_score * weights['dt']
    
    # Determine final classification based on hybrid score
    if total_weighted_score <= 20:
        final_classification = 'safe'
        risk_level = 'Low'
    elif total_weighted_score <= 45:
        final_classification = 'suspicious'
        risk_level = 'Medium'
    else:
        final_classification = 'phishing'
        risk_level = 'High'
    
    # Calculate confidence based on agreement between methods
    confidence_factors = []
    
    # Heuristic confidence
    confidence_factors.append(heuristic_result['confidence'])
    
    # ML confidences if available
    if svm_result[0] not in ['unknown', 'error']:
        confidence_factors.append(svm_result[1])
    if dt_result[0] not in ['unknown', 'error']:
        confidence_factors.append(dt_result[1])
    
    # Average confidence with minimum threshold
    avg_confidence = sum(confidence_factors) / len(confidence_factors)
    final_confidence = max(60, min(95, avg_confidence))
    
    return {
        'classification': final_classification,
        'risk_level': risk_level,
        'confidence': round(final_confidence, 1),
        'hybrid_score': round(total_weighted_score, 1),
        'weights_used': weights,
        'method_agreement': len(set([heuristic_result['classification'], svm_result[0], dt_result[0]])) <= 2
    }

@app.route('/')
def index():
    """Main page with enhanced interface"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_url():
    """Enhanced API endpoint for URL analysis"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'error': 'URL is required',
                'status': 'error'
            }), 400
        
        url = data['url'].strip()
        if not url:
            return jsonify({
                'error': 'URL cannot be empty',
                'status': 'error'
            }), 400
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://', 'ftp://')):
            url = 'http://' + url
        
        print(f"🔍 Analyzing URL: {url}")
        
        # Perform heuristic analysis
        heuristic_result = detector.calculate_risk_score(url)
        if not heuristic_result:
            return jsonify({
                'error': 'Failed to analyze URL',
                'status': 'error'
            }), 500
        
        # Perform ML predictions
        svm_result = predict_with_ml(url, svm_model, "SVM")
        dt_result = predict_with_ml(url, dt_model, "Decision Tree")
        
        # Calculate hybrid score
        hybrid_result = calculate_hybrid_score(heuristic_result, svm_result, dt_result)
        
        # Prepare response
        response = {
            'status': 'success',
            'url': url,
            'analysis': {
                'heuristic': heuristic_result,
                'ml_predictions': {
                    'svm': {
                        'classification': svm_result[0],
                        'confidence': svm_result[1],
                        'details': svm_result[2],
                        'accuracy': svm_accuracy if svm_model else 0
                    },
                    'decision_tree': {
                        'classification': dt_result[0],
                        'confidence': dt_result[1],
                        'details': dt_result[2],
                        'accuracy': dt_accuracy if dt_model else 0
                    }
                },
                'hybrid': hybrid_result,
                'final_result': {
                    'classification': hybrid_result['classification'] if hybrid_result else heuristic_result['classification'],
                    'risk_level': hybrid_result['risk_level'] if hybrid_result else heuristic_result['risk_level'],
                    'confidence': hybrid_result['confidence'] if hybrid_result else heuristic_result['confidence'],
                    'recommendation': get_recommendation(hybrid_result['classification'] if hybrid_result else heuristic_result['classification'])
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'status': 'error'
        }), 500

def get_recommendation(classification):
    """Get security recommendation based on classification"""
    recommendations = {
        'safe': "✅ This URL appears to be safe. However, always remain cautious when entering personal information online.",
        'suspicious': "⚠️ This URL shows suspicious characteristics. Exercise extreme caution. Verify the website's legitimacy before proceeding.",
        'phishing': "🚨 This URL is likely a phishing attempt. DO NOT enter any personal information. Block this URL and report it if possible."
    }
    return recommendations.get(classification, "❓ Unable to determine safety. Proceed with extreme caution.")

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    """Batch URL analysis endpoint"""
    try:
        data = request.get_json()
        if not data or 'urls' not in data:
            return jsonify({
                'error': 'URLs list is required',
                'status': 'error'
            }), 400
        
        urls = data['urls']
        if not isinstance(urls, list):
            return jsonify({
                'error': 'URLs must be provided as a list',
                'status': 'error'
            }), 400
        
        if len(urls) > 50:  # Limit batch size
            return jsonify({
                'error': 'Maximum 50 URLs allowed per batch',
                'status': 'error'
            }), 400
        
        results = []
        for i, url in enumerate(urls):
            if not url or not isinstance(url, str):
                continue
                
            url = url.strip()
            if not url.startswith(('http://', 'https://', 'ftp://')):
                url = 'http://' + url
            
            try:
                heuristic_result = detector.calculate_risk_score(url)
                if heuristic_result:
                    svm_result = predict_with_ml(url, svm_model, "SVM")
                    dt_result = predict_with_ml(url, dt_model, "Decision Tree")
                    hybrid_result = calculate_hybrid_score(heuristic_result, svm_result, dt_result)
                    
                    results.append({
                        'url': url,
                        'classification': hybrid_result['classification'] if hybrid_result else heuristic_result['classification'],
                        'risk_level': hybrid_result['risk_level'] if hybrid_result else heuristic_result['risk_level'],
                        'confidence': hybrid_result['confidence'] if hybrid_result else heuristic_result['confidence'],
                        'score': hybrid_result['hybrid_score'] if hybrid_result else heuristic_result['total_score']
                    })
                else:
                    results.append({
                        'url': url,
                        'classification': 'error',
                        'risk_level': 'Unknown',
                        'confidence': 0,
                        'score': 0,
                        'error': 'Analysis failed'
                    })
            except Exception as e:
                results.append({
                    'url': url,
                    'classification': 'error',
                    'risk_level': 'Unknown',
                    'confidence': 0,
                    'score': 0,
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total_analyzed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Batch analysis failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics and model information"""
    try:
        stats = {
            'system_info': {
                'version': '2.0.0',
                'last_updated': '2024-12-12',
                'features': [
                    'Enhanced Heuristic Analysis',
                    'Machine Learning Integration',
                    'Hybrid Scoring System',
                    'Batch Processing',
                    'Real-time Analysis'
                ]
            },
            'datasets': {
                'whitelist_domains': len(detector.whitelist_domains),
                'phishing_domains': len(detector.phishing_domains),
                'legitimate_domains': len(detector.legitimate_domains)
            },
            'ml_models': {
                'svm': {
                    'available': svm_model is not None,
                    'accuracy': round(svm_accuracy, 3) if svm_model else 0,
                    'confidence_threshold': CONFIG['model_confidence_threshold']
                },
                'decision_tree': {
                    'available': dt_model is not None,
                    'accuracy': round(dt_accuracy, 3) if dt_model else 0
                }
            },
            'analysis_weights': detector.weights,
            'classification_thresholds': detector.thresholds,
            'configuration': {
                key: value for key, value in CONFIG.items() 
                if key not in ['max_dataset_rows']  # Hide internal configs
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get stats: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_models_loaded': {
            'svm': svm_model is not None,
            'decision_tree': dt_model is not None
        },
        'datasets_loaded': {
            'whitelist': len(detector.whitelist_domains) > 0,
            'phishing': len(detector.phishing_domains) > 0
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({
        'error': 'Bad request',
        'status': 'error'
    }), 400

if __name__ == '__main__':
    print("🚀 Enhanced Phishing Detection System v2.0")
    print("=" * 50)
    print(f"🔧 Configuration:")
    print(f"   - Max dataset rows: {CONFIG['max_dataset_rows']}")
    print(f"   - Test size: {CONFIG['test_size']}")
    print(f"   - Random state: {CONFIG['random_state']}")
    print(f"   - Model confidence threshold: {CONFIG['model_confidence_threshold']}")
    print(f"   - Max features: {CONFIG['max_features']}")
    print("=" * 50)
    print(f"📊 Datasets loaded:")
    print(f"   - Whitelist domains: {len(detector.whitelist_domains)}")
    print(f"   - Phishing domains: {len(detector.phishing_domains)}")
    print(f"   - Legitimate domains: {len(detector.legitimate_domains)}")
    print("=" * 50)
    print(f"🤖 ML Models:")
    print(f"   - SVM: {'✅ Ready' if svm_model else '❌ Not available'}")
    print(f"   - Decision Tree: {'✅ Ready' if dt_model else '❌ Not available'}")
    if svm_model:
        print(f"   - SVM Accuracy: {svm_accuracy:.3f}")
    if dt_model:
        print(f"   - DT Accuracy: {dt_accuracy:.3f}")
    print("=" * 50)
    print("🌐 Starting Flask server...")
    print("📡 API Endpoints:")
    print("   - POST /api/analyze - Single URL analysis")
    print("   - POST /api/batch - Batch URL analysis")
    print("   - GET /api/stats - System statistics")
    print("   - GET /api/health - Health check")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)