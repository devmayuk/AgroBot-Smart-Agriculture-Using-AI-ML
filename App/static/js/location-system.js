// Comprehensive Location System for AgroBot
// GPS Detection + Manual Indian States/Cities Database

class AgroLocation {
    constructor() {
        this.currentLocation = null;
        this.isDetecting = false;
        this.callbacks = {};
        this.init();
    }

    // Indian States and Cities Database
    indianStates = {
        "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Kurnool", "Tirupati", "Rajahmundry", "Kakinada", "Anantapur", "Eluru", "Ongole", "Nizamabad", "Vizianagaram", "Srikakulam", "Proddatur"],
        "Arunachal Pradesh": ["Itanagar", "Tawang", "Ziro", "Bomdila", "Pasighat", "Changlang", "Tezu", "Along", "Anini", "Khonsa", "Yingkiong", "Daporijo", "Basar", "Aalo", "Pangin"],
        "Assam": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat", "Nagaon", "Tinsukia", "Tezpur", "Bongaigaon", "Karimganj", "Sivasagar", "Lakhimpur", "Dhubri", "Goalpara", "Barpeta", "Morigaon"],
        "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Purnia", "Darbhanga", "Bihar Sharif", "Ara", "Begusarai", "Katihar", "Munger", "Chhapra", "Danapur", "Dehri-on-Sone", "Sasaram"],
        "Chhattisgarh": ["Raipur", "Bhilai", "Durg", "Bilaspur", "Korba", "Rajnandgaon", "Raigarh", "Jagdalpur", "Ambikapur", "Dhamtari", "Mahasamund", "Kawardha", "Bastar", "Kanker", "Balod"],
        "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda", "Bicholim", "Curchorem", "Sanquelim", "Cortalim", "Quepem", "Sao Jose de Areal", "Chorao", "Benaulim", "Candolim", "Arpora"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Junagadh", "Gandhinagar", "Anand", "Nadiad", "Porbandar", "Veraval", "Bhuj", "Gandhidham", "Mehsana"],
        "Haryana": ["Gurgaon", "Faridabad", "Panipat", "Ambala", "Karnal", "Sonipat", "Rohtak", "Hisar", "Bhiwani", "Sirsa", "Yamunanagar", "Kurukshetra", "Jind", "Kaithal", "Rewari"],
        "Himachal Pradesh": ["Shimla", "Solan", "Dharamshala", "Mandi", "Palampur", "Baddi", "Kullu", "Manali", "Una", "Sundarnagar", "Nahan", "Chamba", "Hamirpur", "Nurpur", "Jogindernagar"],
        "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Deoghar", "Hazaribagh", "Giridih", "Ramgarh", "Chaibasa", "Medininagar", "Chakradharpur", "Phusro", "Patratu", "Muri", "Tatanagar"],
        "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi-Dharwad", "Belagavi", "Gulbarga", "Davanagere", "Bellary", "Bijapur", "Shivamogga", "Tumakuru", "Udupi", "Raichur", "Bidar", "Hassan"],
        "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Kollam", "Thrissur", "Palakkad", "Alappuzha", "Malappuram", "Kannur", "Kasaragod", "Kottayam", "Idukki", "Wayanad", "Ernakulam", "Pathanamthitta"],
        "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Sagar", "Dewas", "Satna", "Ratlam", "Rewa", "Murwara", "Singrauli", "Burhanpur", "Khandwa", "Bhind"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik", "Aurangabad", "Solapur", "Amravati", "Navi Mumbai", "Kolhapur", "Sangli", "Malegaon", "Jalgaon", "Latur", "Dhule"],
        "Manipur": ["Imphal", "Thoubal", "Churachandpur", "Bishnupur", "Kakching", "Ukhrul", "Senapati", "Tamenglong", "None", "Jiribam", "Moreh", "Moirang", "Mayang Imphal", "Wangjing", "Andro"],
        "Meghalaya": ["Shillong", "Tura", "Nongstoin", "Baghmara", "Jowai", "Williamnagar", "Resubelpara", "Ampati", "Mairang", "Mawkyrwat", "Mawsynram", "Shella", "Rongram", "Khliehriat", "Mawsram"],
        "Mizoram": ["Aizawl", "Lunglei", "Saiha", "Champhai", "Kolasib", "Serchhip", "Mamit", "Lawngtlai", "Saitual", "Hnahthial", "Khawzawl", "Thenzawl", "Tawipui", "Biate", "Vairengte"],
        "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Tuensang", "Wokha", "Zunheboto", "Phek", "Kiphire", "Longleng", "Mon", "Peren", "Tseminyu", "Jalukie", "Chozuba", "Phek"],
        "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Puri", "Sambalpur", "Berhampur", "Rourkela", "Baleshwar", "Baripada", "Balangir", "Bhadrak", "Jajpur", "Jagatsinghpur", "Kendrapara", "Keonjhar"],
        "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Mohali", "Batala", "Pathankot", "Firozpur", "Abohar", "Moga", "Khanna", "Barnala", "Fazilka", "Kapurthala"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer", "Bikaner", "Bhilwara", "Alwar", "Sikar", "Pali", "Bharatpur", "Ganganagar", "Churu", "Tonk", "Barmer"],
        "Sikkim": ["Gangtok", "Namchi", "Mangan", "Gyalshing", "Rangpo", "Singtam", "Jorethang", "Nayabazar", "Rangpo", "Pakyong", "Rongli", "Lachung", "Lachen", "Mangan", "Yuksom"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Erode", "Tiruppur", "Vellore", "Thoothukudi", "Dindigul", "Thanjavur", "Ranipet", "Nagercoil", "Sivakasi", "Karur"],
        "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam", "Ramagundam", "Mahbubnagar", "Nalgonda", "Siddipet", "Adilabad", "Medak", "Jagtial", "Kamareddy", "Sircilla", "Mancherial"],
        "Tripura": ["Agartala", "Udaipur", "Dharmanagar", "Pratapgarh", "Kailashahar", "Belonia", "Sabroom", "Khowai", "Teliamura", "Amarpur", "Ranirbazar", "Jampui Hills", "Sepahijala", "Gandacherra", "Kumarghat"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Ghaziabad", "Agra", "Varanasi", "Meerut", "Allahabad", "Bareilly", "Aligarh", "Moradabad", "Saharanpur", "Gorakhpur", "Noida", "Firozabad", "Jhansi"],
        "Uttarakhand": ["Dehradun", "Haridwar", "Roorkee", "Haldwani", "Rishikesh", "Kashipur", "Rudrapur", "Kashipur", "Roorkee", "Nainital", "Mussoorie", "Almora", "Pithoragarh", "Bageshwar", "Champawat"],
        "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri", "Burdwan", "Baharampur", "Kharagpur", "Shantiniketan", "Habra", "Barrackpore", "Serampore", "Chandannagar", "Madhyamgram", "Barasat"],
        "Andaman and Nicobar Islands": ["Port Blair", "Car Nicobar", "Great Nicobar", "Havelock Island", "Neil Island", "Long Island", "Rangat", "Mayabunder", "Diglipur", "Little Andaman"],
        "Chandigarh": ["Chandigarh"],
        "Dadra and Nagar Haveli and Daman and Diu": ["Silvassa", "Daman", "Diu", "Vapi", "Bhilad", "Union Territory"],
        "Delhi": ["New Delhi", "Central Delhi", "North Delhi", "South Delhi", "East Delhi", "West Delhi", "North West Delhi", "North East Delhi", "South West Delhi", "New Delhi District"],
        "Jammu and Kashmir": ["Srinagar", "Jammu", "Anantnag", "Baramulla", "Sopore", "Kathua", "Udhampur", "Pulwama", "Kupwara", "Rajouri", "Poonch", "Kargil", "Leh", "Ganderbal", "Kulgam"],
        "Ladakh": ["Leh", "Kargil", "Nubra", "Zanskar", "Changthang", "Sham Valley", "Pangong", "Turtuk", "Diskit", "Hunder"],
        "Lakshadweep": ["Kavaratti", "Agatti", "Bangaram", "Minicoy", "Kadmat", "Kalpeni", "Andrott", "Amini", "Bitra", "Chetlat", "Kiltan", "Suheli"],
        "Pondicherry": ["Puducherry", "Karaikal", "Mahe", "Yanam"]
    };

    init() {
        this.setupLocationDetection();
        this.setupManualLocation();
    }

    // GPS Location Detection
    async detectGPSLocation() {
        if (!navigator.geolocation) {
            throw new Error('Geolocation is not supported by this browser');
        }

        this.isDetecting = true;
        this.showLoadingState();

        return new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(
                async (position) => {
                    this.isDetecting = false;
                    const location = await this.reverseGeocode(position.coords.latitude, position.coords.longitude);
                    this.currentLocation = location;
                    this.hideLoadingState();
                    this.onLocationDetected(location);
                    resolve(location);
                },
                (error) => {
                    this.isDetecting = false;
                    this.hideLoadingState();
                    this.handleLocationError(error);
                    reject(error);
                },
                {
                    enableHighAccuracy: true,
                    timeout: 15000,
                    maximumAge: 0
                }
            );
        });
    }

    // Reverse Geocoding using OpenStreetMap Nominatim API
    async reverseGeocode(lat, lon) {
        try {
            const response = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10&addressdetails=1&accept-language=en-US,en;q=0.9`);
            const data = await response.json();

            if (data && data.address) {
                const address = data.address;

                // Find Indian state
                let state = address.state || '';
                let city = address.city || address.town || address.village || '';

                // If state not found in address, try to detect from components
                if (!state && address.state_district) {
                    state = address.state_district;
                }

                // Map various Indian state names
                const stateMappings = {
                    'Tamil Nadu': 'Tamil Nadu',
                    'Tamilnadu': 'Tamil Nadu',
                    'Uttar Pradesh': 'Uttar Pradesh',
                    'Uttarpradesh': 'Uttar Pradesh',
                    'Madhya Pradesh': 'Madhya Pradesh',
                    'Madhyapradesh': 'Madhya Pradesh',
                    'Andhra Pradesh': 'Andhra Pradesh',
                    'Andhrapradesh': 'Andhra Pradesh',
                    'Arunachal Pradesh': 'Arunachal Pradesh',
                    'Arunachalpradesh': 'Arunachal Pradesh',
                    'Himachal Pradesh': 'Himachal Pradesh',
                    'Himachalpradesh': 'Himachal Pradesh',
                    'Jammu and Kashmir': 'Jammu and Kashmir',
                    'Jammu & Kashmir': 'Jammu and Kashmir'
                };

                for (const [key, value] of Object.entries(stateMappings)) {
                    if (address[key]) {
                        state = value;
                        break;
                    }
                }

                // Default to major cities if nothing found
                if (!city && !state) {
                    // Try to identify region from coordinates
                    const region = this.identifyRegionByCoordinates(lat, lon);
                    return region;
                }

                return {
                    latitude: lat,
                    longitude: lon,
                    city: city,
                    state: state || 'Unknown',
                    country: address.country || 'India',
                    display_name: data.display_name,
                    accuracy: 'GPS'
                };
            }
        } catch (error) {
            console.error('Reverse geocoding failed:', error);
            return this.identifyRegionByCoordinates(lat, lon);
        }
    }

    // Identify region by coordinates (fallback method)
    identifyRegionByCoordinates(lat, lon) {
        // Rough coordinate boundaries for major Indian cities
        const cities = [
            { name: 'New Delhi', state: 'Delhi', lat: 28.6139, lon: 77.2090, radius: 0.5 },
            { name: 'Mumbai', state: 'Maharashtra', lat: 19.0760, lon: 72.8777, radius: 0.5 },
            { name: 'Bengaluru', state: 'Karnataka', lat: 12.9716, lon: 77.5946, radius: 0.5 },
            { name: 'Chennai', state: 'Tamil Nadu', lat: 13.0827, lon: 80.2707, radius: 0.5 },
            { name: 'Kolkata', state: 'West Bengal', lat: 22.5726, lon: 88.3639, radius: 0.5 },
            { name: 'Hyderabad', state: 'Telangana', lat: 17.3850, lon: 78.4867, radius: 0.5 },
            { name: 'Pune', state: 'Maharashtra', lat: 18.5204, lon: 73.8567, radius: 0.5 },
            { name: 'Ahmedabad', state: 'Gujarat', lat: 23.0225, lon: 72.5714, radius: 0.5 },
            { name: 'Jaipur', state: 'Rajasthan', lat: 26.9124, lon: 75.7873, radius: 0.5 },
            { name: 'Lucknow', state: 'Uttar Pradesh', lat: 26.8467, lon: 80.9462, radius: 0.5 }
        ];

        let closestCity = null;
        let minDistance = Infinity;

        cities.forEach(city => {
            const distance = this.calculateDistance(lat, lon, city.lat, city.lon);
            if (distance < minDistance) {
                minDistance = distance;
                closestCity = city;
            }
        });

        if (closestCity && minDistance < 2) { // Within ~200km
            return {
                latitude: lat,
                longitude: lon,
                city: closestCity.name,
                state: closestCity.state,
                country: 'India',
                display_name: `${closestCity.name}, ${closestCity.state}, India`,
                accuracy: 'Approximate'
            };
        }

        // Default to a major city if no close match
        return {
            latitude: lat,
            longitude: lon,
            city: 'New Delhi',
            state: 'Delhi',
            country: 'India',
            display_name: 'New Delhi, Delhi, India',
            accuracy: 'Default'
        };
    }

    calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Earth's radius in km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    handleLocationError(error) {
        let message = '';
        switch(error.code) {
            case error.PERMISSION_DENIED:
                message = 'Location access denied. Please select your location manually.';
                break;
            case error.POSITION_UNAVAILABLE:
                message = 'Location information unavailable. Please select your location manually.';
                break;
            case error.TIMEOUT:
                message = 'Location request timed out. Please select your location manually.';
                break;
            default:
                message = 'An error occurred while detecting location. Please select manually.';
                break;
        }
        this.showManualSelection(message);
    }

    // Manual Location Selection
    setupManualLocation() {
        this.populateStates();
    }

    populateStates() {
        const stateSelect = document.getElementById('locationState');
        if (stateSelect) {
            stateSelect.innerHTML = '<option value="">Select State</option>';

            Object.keys(this.indianStates).sort().forEach(state => {
                const option = document.createElement('option');
                option.value = state;
                option.textContent = state;
                stateSelect.appendChild(option);
            });
        }
    }

    populateCities(state) {
        const citySelect = document.getElementById('locationCity');
        if (citySelect && state && this.indianStates[state]) {
            citySelect.innerHTML = '<option value="">Select City</option>';

            // Sort cities with popular ones first
            const popularCities = this.getPopularCities(state);
            const otherCities = this.indianStates[state].filter(city => !popularCities.includes(city));

            [...popularCities, ...otherCities].forEach(city => {
                const option = document.createElement('option');
                option.value = city;
                option.textContent = city;
                citySelect.appendChild(option);
            });
        }
    }

    getPopularCities(state) {
        const popularCities = {
            'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik'],
            'Karnataka': ['Bengaluru', 'Mysuru', 'Mangaluru', 'Hubballi'],
            'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli'],
            'Delhi': ['New Delhi', 'North Delhi', 'South Delhi'],
            'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot'],
            'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi'],
            'West Bengal': ['Kolkata', 'Howrah', 'Siliguri', 'Durgapur'],
            'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota'],
            'Andhra Pradesh': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore'],
            'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar']
        };
        return popularCities[state] || [];
    }

    // UI Methods
    showLoadingState() {
        const container = document.getElementById('locationContainer');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <div class="spinner-border text-agro-primary mb-3" role="status">
                        <span class="visually-hidden">Detecting location...</span>
                    </div>
                    <h5 class="text-agro-primary">
                        <i class="fas fa-location-dot me-2"></i>
                        Detecting your location...
                    </h5>
                    <p class="text-muted">Please allow location access for accurate weather data</p>
                </div>
            `;
        }
    }

    hideLoadingState() {
        // This will be called when location is detected or error occurs
    }

    showManualSelection(errorMessage = '') {
        const container = document.getElementById('locationContainer');
        if (container) {
            container.innerHTML = `
                <div class="location-manual">
                    ${errorMessage ? `<div class="alert alert-warning mb-3">
                        <i class="fas fa-exclamation-triangle me-2"></i>${errorMessage}
                    </div>` : ''}

                    <div class="d-flex gap-3 mb-3">
                        <button class="btn-agro flex-fill" onclick="agroLocation.detectGPSLocation()">
                            <i class="fas fa-location-dot me-2"></i>
                            Try GPS Again
                        </button>
                        <button class="btn btn-outline-secondary flex-fill" onclick="agroLocation.showManualForm()">
                            <i class="fas fa-hand-pointer me-2"></i>
                            Select Manually
                        </button>
                    </div>

                    <div class="popular-cities-quick mb-3">
                        <p class="text-muted mb-2">Quick Select:</p>
                        <div class="d-flex flex-wrap gap-2">
                            <button class="btn btn-sm btn-outline-agro" onclick="agroLocation.quickSelect('Mumbai', 'Maharashtra')">
                                <i class="fas fa-city me-1"></i>Mumbai
                            </button>
                            <button class="btn btn-sm btn-outline-agro" onclick="agroLocation.quickSelect('Delhi', 'Delhi')">
                                <i class="fas fa-city me-1"></i>Delhi
                            </button>
                            <button class="btn btn-sm btn-outline-agro" onclick="agroLocation.quickSelect('Bengaluru', 'Karnataka')">
                                <i class="fas fa-city me-1"></i>Bengaluru
                            </button>
                            <button class="btn btn-sm btn-outline-agro" onclick="agroLocation.quickSelect('Chennai', 'Tamil Nadu')">
                                <i class="fas fa-city me-1"></i>Chennai
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    showManualForm() {
        const container = document.getElementById('locationContainer');
        if (container) {
            container.innerHTML = `
                <div class="location-form">
                    <h6 class="mb-3">
                        <i class="fas fa-map-marker-alt me-2 text-agro-primary"></i>
                        Select Your Location
                    </h6>

                    <div class="mb-3">
                        <label class="form-label">State</label>
                        <select id="locationState" class="form-control" onchange="agroLocation.onStateChange()">
                            <option value="">Select State</option>
                        </select>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">City</label>
                        <select id="locationCity" class="form-control">
                            <option value="">Select City</option>
                        </select>
                    </div>

                    <div class="input-group mb-3">
                        <input type="text" id="citySearch" class="form-control" placeholder="Search city..."
                               onkeyup="agroLocation.searchCities(this.value)">
                        <span class="input-group-text">
                            <i class="fas fa-search"></i>
                        </span>
                    </div>

                    <button class="btn-agro w-100" onclick="agroLocation.setManualLocation()">
                        <i class="fas fa-check me-2"></i>
                        Set Location
                    </button>
                </div>
            `;
            this.populateStates();
        }
    }

    quickSelect(city, state) {
        this.currentLocation = {
            city: city,
            state: state,
            country: 'India',
            accuracy: 'Manual'
        };
        this.onLocationDetected(this.currentLocation);
    }

    onStateChange() {
        const stateSelect = document.getElementById('locationState');
        if (stateSelect) {
            this.populateCities(stateSelect.value);
        }
    }

    searchCities(query) {
        const citySelect = document.getElementById('locationCity');
        if (!citySelect || !query) return;

        const stateSelect = document.getElementById('locationState');
        const state = stateSelect?.value;

        if (!state) return;

        const cities = this.indianStates[state];
        const filteredCities = cities.filter(city =>
            city.toLowerCase().includes(query.toLowerCase())
        );

        citySelect.innerHTML = '<option value="">Select City</option>';
        filteredCities.forEach(city => {
            const option = document.createElement('option');
            option.value = city;
            option.textContent = city;
            citySelect.appendChild(option);
        });
    }

    setManualLocation() {
        const stateSelect = document.getElementById('locationState');
        const citySelect = document.getElementById('locationCity');

        if (!stateSelect.value || !citySelect.value) {
            alert('Please select both state and city');
            return;
        }

        this.currentLocation = {
            city: citySelect.value,
            state: stateSelect.value,
            country: 'India',
            accuracy: 'Manual'
        };

        this.onLocationDetected(this.currentLocation);
    }

    onLocationDetected(location) {
        // Update form fields
        const cityInput = document.getElementById('city');
        const stateInput = document.getElementById('stt');

        if (cityInput) cityInput.value = location.city;
        if (stateInput) {
            // Try to set the state dropdown
            for (let option of stateInput.options) {
                if (option.text === location.state) {
                    stateInput.value = option.value;
                    break;
                }
            }
        }

        // Trigger cities.js to update city dropdown
        if (typeof print_city === 'function') {
            const stateIndex = Array.from(stateInput.options).findIndex(opt => opt.text === location.state);
            if (stateIndex > 0) {
                print_city('state', stateIndex);
                setTimeout(() => {
                    const cityDropdown = document.getElementById('state');
                    if (cityDropdown) {
                        for (let option of cityDropdown.options) {
                            if (option.text === location.city) {
                                cityDropdown.value = option.value;
                                break;
                            }
                        }
                    }
                }, 100);
            }
        }

        // Update display
        this.updateLocationDisplay(location);

        // Trigger callback
        if (this.callbacks.onLocationDetected) {
            this.callbacks.onLocationDetected(location);
        }
    }

    updateLocationDisplay(location) {
        const displayElement = document.getElementById('locationDisplay');
        if (displayElement) {
            displayElement.innerHTML = `
                <div class="alert alert-success mb-3">
                    <i class="fas fa-map-marker-alt me-2"></i>
                    <strong>Location detected:</strong> ${location.city}, ${location.state}
                    <br><small class="text-muted">Accuracy: ${location.accuracy}</small>
                </div>
            `;
        }
    }

    setupLocationDetection() {
        // This will be called when the page loads
        this.detectGPSLocation().catch(() => {
            // Auto-fallback to manual selection
            this.showManualSelection();
        });
    }

    // Public methods
    getCurrentLocation() {
        return this.currentLocation;
    }

    onLocationChange(callback) {
        this.callbacks.onLocationDetected = callback;
    }
}

// Initialize the location system
const agroLocation = new AgroLocation();

// Export for global access
window.agroLocation = agroLocation;