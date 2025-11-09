# Setup script for the time series analysis project

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "models/checkpoints",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created {directory}")
    
    print("✅ All directories created!")

def run_tests():
    """Run unit tests."""
    print("🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
        print("✅ All tests passed!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Some tests failed: {e}")
    except FileNotFoundError:
        print("⚠️ pytest not found, skipping tests")

def main():
    """Main setup function."""
    print("🚀 Setting up Time Series Analysis Project")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create directories
    create_directories()
    
    # Run tests
    run_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\n💡 Next steps:")
    print("  - Run 'python example.py' for a quick demo")
    print("  - Run 'streamlit run app.py' for the web interface")
    print("  - Open notebooks/ for detailed analysis")

if __name__ == "__main__":
    main()
