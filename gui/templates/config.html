<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MacroEconVue Configuration</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
</head>
<body class="bg-gray-50 min-h-screen">
    <!-- Header -->
    <header class="bg-white shadow">
        <div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <div class="flex items-center">
                <span class="material-icons text-blue-600 mr-3 text-3xl">analytics</span>
                <h1 class="text-3xl font-bold text-gray-900">MacroEconVue Configuration</h1>
            </div>
            <p class="mt-2 text-gray-600">Edit simulation parameters and agent settings</p>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <!-- Notification Area -->
        <div id="notification-area" class="mb-4"></div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Configuration Section -->
            <div class="bg-white shadow rounded-lg">
                <div class="px-6 py-5 border-b border-gray-200">
                    <h2 class="text-xl font-semibold text-gray-900">
                        <span class="material-icons mr-2 align-middle">settings</span>
                        Simulation Configuration
                    </h2>
                </div>
                <div class="p-6">
                    <form id="config-form">
                        <!-- Simulation Settings -->
                        <div class="mb-6">
                            <h3 class="text-lg font-medium text-gray-900 mb-4">Simulation Settings</h3>
                            <div class="space-y-4">
                                <div>
                                    <label for="periods" class="block text-sm font-medium text-gray-700">Periods</label>
                                    <input type="number" id="periods" name="periods" value="{{ config.simulation.periods }}" 
                                           class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500">
                                </div>
                                
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-2">Agents per Type</label>
                                    <div class="grid grid-cols-2 gap-3">
                                        {% for agent_type, count in config.simulation.agents_per_type.items() %}
                                        <div>
                                            <label for="agent-{{ agent_type }}" class="block text-xs font-medium text-gray-600">
                                                {{ agent_type.replace('_', ' ').title() }}
                                            </label>
                                            <input type="number" id="agent-{{ agent_type }}" name="agent-{{ agent_type }}" 
                                                   value="{{ count }}" min="0" 
                                                   class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                                        </div>
                                        {% endfor %}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Economics Settings -->
                        <div class="mb-6 border-t pt-6">
                            <h3 class="text-lg font-medium text-gray-900 mb-4">Economics Settings</h3>
                            <div class="space-y-4">
                                <div>
                                    <label for="interest-rate" class="block text-sm font-medium text-gray-700">Interest Rate</label>
                                    <input type="number" id="interest-rate" name="interest-rate" 
                                           value="{{ config.economics.interest_rate }}" step="0.001" min="0" max="1"
                                           class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500">
                                    <p class="mt-1 text-xs text-gray-500">As decimal (e.g., 0.04 = 4%)</p>
                                </div>
                                
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-2">Discretionary Goods Prices ($)</label>
                                    <div class="grid grid-cols-2 gap-3">
                                        {% for good, price in config.economics.discretionary_goods.items() %}
                                        <div>
                                            <label for="good-{{ good }}" class="block text-xs font-medium text-gray-600">
                                                {{ good.title() }}
                                            </label>
                                            <input type="number" id="good-{{ good }}" name="good-{{ good }}" 
                                                   value="{{ price }}" step="0.01" min="0"
                                                   class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                                        </div>
                                        {% endfor %}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- LLM Settings -->
                        <div class="border-t pt-6">
                            <h3 class="text-lg font-medium text-gray-900 mb-4">LLM Settings</h3>
                            <div class="space-y-4">
                                <div>
                                    <label for="model-name" class="block text-sm font-medium text-gray-700">Model Name</label>
                                    <input type="text" id="model-name" name="model-name" 
                                           value="{{ config.llm.model_name }}"
                                           class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500">
                                </div>
                                <div class="grid grid-cols-2 gap-3">
                                    <div>
                                        <label for="temperature" class="block text-sm font-medium text-gray-700">Temperature</label>
                                        <input type="number" id="temperature" name="temperature" 
                                               value="{{ config.llm.temperature }}" step="0.1" min="0" max="2"
                                               class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500">
                                    </div>
                                    <div>
                                        <label for="max-tokens" class="block text-sm font-medium text-gray-700">Max Tokens</label>
                                        <input type="number" id="max-tokens" name="max-tokens" 
                                               value="{{ config.llm.max_tokens }}" min="1" max="4000"
                                               class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500">
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Save Button -->
                        <div class="mt-8 pt-6 border-t">
                            <button type="submit" 
                                    class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition duration-150">
                                Save Configuration
                            </button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Agents Section -->
            <div class="bg-white shadow rounded-lg">
                <div class="px-6 py-5 border-b border-gray-200">
                    <div class="flex justify-between items-center">
                        <h2 class="text-xl font-semibold text-gray-900">
                            <span class="material-icons mr-2 align-middle">people</span>
                            Agent Types
                        </h2>
                        <button onclick="addNewAgent()" 
                                class="bg-green-600 hover:bg-green-700 text-white text-sm font-medium py-1 px-3 rounded-md transition duration-150">
                            <span class="material-icons mr-1 text-sm">add</span>
                            Add Agent
                        </button>
                    </div>
                </div>
                <div class="p-6">
                    <div class="space-y-4" id="agents-container">
                        {% for agent in agents %}
                        <div class="agent-card border border-gray-200 rounded-lg p-4" data-agent-index="{{ loop.index0 }}">
                            <div class="grid grid-cols-2 gap-3">
                                <div class="col-span-2">
                                    <label class="block text-sm font-medium text-gray-700">Agent Type</label>
                                    <input type="text" value="{{ agent.agent_type }}" 
                                           class="agent-type mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Income ($)</label>
                                    <input type="number" value="{{ agent.income }}" step="0.01" min="0"
                                           class="agent-income mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Housing ($)</label>
                                    <input type="number" value="{{ agent.housing }}" step="0.01" min="0"
                                           class="agent-housing mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Insurance ($)</label>
                                    <input type="number" value="{{ agent.insurance }}" step="0.01" min="0"
                                           class="agent-insurance mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Healthcare ($)</label>
                                    <input type="number" value="{{ agent.healthcare }}" step="0.01" min="0"
                                           class="agent-healthcare mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Repair ($)</label>
                                    <input type="number" value="{{ agent.repair }}" step="0.01" min="0"
                                           class="agent-repair mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                                </div>
                                <div class="col-span-2 flex justify-end">
                                    <button onclick="deleteAgent(this)" 
                                            class="text-red-600 hover:text-red-900 bg-red-100 hover:bg-red-200 p-2 rounded text-sm">
                                        <span class="material-icons text-sm">delete</span>
                                        Delete
                                    </button>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>

                    <!-- Save Agents Button -->
                    <div class="mt-6 pt-6 border-t">
                        <button onclick="saveAgents()" 
                                class="w-full bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-4 rounded-md transition duration-150">
                            Save Agents
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- JavaScript -->
    <script>
        // Utility functions
        function showNotification(message, type = 'info') {
            const notificationArea = document.getElementById('notification-area');
            const bgColor = type === 'error' ? 'bg-red-100 border-red-400 text-red-700' : 'bg-green-100 border-green-400 text-green-700';
            
            const notification = document.createElement('div');
            notification.className = `${bgColor} border px-4 py-3 rounded mb-2`;
            notification.innerHTML = `
                <div class="flex justify-between items-center">
                    <span>${message}</span>
                    <button onclick="this.parentElement.parentElement.remove()" class="text-sm underline">Close</button>
                </div>
            `;
            
            notificationArea.appendChild(notification);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 5000);
        }

        async function apiCall(url, method = 'GET', data = null) {
            try {
                const options = {
                    method: method,
                    headers: {
                        'Content-Type': 'application/json',
                    },
                };
                
                if (data) {
                    options.body = JSON.stringify(data);
                }
                
                const response = await fetch(url, options);
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.message || 'API call failed');
                }
                
                return result;
            } catch (error) {
                console.error('API Error:', error);
                throw error;
            }
        }

        // Configuration form handler
        document.getElementById('config-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const configData = {
                simulation: {
                    periods: parseInt(formData.get('periods')),
                    agents_per_type: {
                        young_professional: parseInt(formData.get('agent-young_professional')),
                        family: parseInt(formData.get('agent-family')),
                        retiree: parseInt(formData.get('agent-retiree')),
                        student: parseInt(formData.get('agent-student')),
                        low_income: parseInt(formData.get('agent-low_income'))
                    }
                },
                economics: {
                    interest_rate: parseFloat(formData.get('interest-rate')),
                    discretionary_goods: {
                        entertainment: parseFloat(formData.get('good-entertainment')),
                        travel: parseFloat(formData.get('good-travel'))
                    },
                    fixed_costs: ["housing", "insurance"],
                    variable_costs: ["healthcare", "repair"]
                },
                llm: {
                    model_name: formData.get('model-name'),
                    temperature: parseFloat(formData.get('temperature')),
                    max_tokens: parseInt(formData.get('max-tokens'))
                }
            };
            
            try {
                const result = await apiCall('/api/config', 'POST', configData);
                if (result.success) {
                    showNotification('Configuration saved successfully!', 'success');
                } else {
                    showNotification('Failed to save configuration: ' + result.message, 'error');
                }
            } catch (error) {
                showNotification('Error saving configuration: ' + error.message, 'error');
            }
        });

        // Agent management functions
        function addNewAgent() {
            const container = document.getElementById('agents-container');
            const newIndex = container.children.length;
            
            const newAgentDiv = document.createElement('div');
            newAgentDiv.className = 'agent-card border border-gray-200 rounded-lg p-4';
            newAgentDiv.setAttribute('data-agent-index', newIndex);
            newAgentDiv.innerHTML = `
                <div class="grid grid-cols-2 gap-3">
                    <div class="col-span-2">
                        <label class="block text-sm font-medium text-gray-700">Agent Type</label>
                        <input type="text" value="new_agent" 
                               class="agent-type mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Income ($)</label>
                        <input type="number" value="1000.00" step="0.01" min="0"
                               class="agent-income mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Housing ($)</label>
                        <input type="number" value="200.00" step="0.01" min="0"
                               class="agent-housing mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Insurance ($)</label>
                        <input type="number" value="75.00" step="0.01" min="0"
                               class="agent-insurance mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Healthcare ($)</label>
                        <input type="number" value="100.00" step="0.01" min="0"
                               class="agent-healthcare mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Repair ($)</label>
                        <input type="number" value="40.00" step="0.01" min="0"
                               class="agent-repair mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm">
                    </div>
                    <div class="col-span-2 flex justify-end">
                        <button onclick="deleteAgent(this)" 
                                class="text-red-600 hover:text-red-900 bg-red-100 hover:bg-red-200 p-2 rounded text-sm">
                            <span class="material-icons text-sm">delete</span>
                            Delete
                        </button>
                    </div>
                </div>
            `;
            
            container.appendChild(newAgentDiv);
        }

        function deleteAgent(button) {
            if (confirm('Are you sure you want to delete this agent type?')) {
                button.closest('.agent-card').remove();
            }
        }

        async function saveAgents() {
            const agentCards = document.querySelectorAll('.agent-card');
            const agents = [];
            
            for (const card of agentCards) {
                const agent = {
                    agent_type: card.querySelector('.agent-type').value,
                    income: parseFloat(card.querySelector('.agent-income').value).toFixed(2),
                    housing: parseFloat(card.querySelector('.agent-housing').value).toFixed(2),
                    insurance: parseFloat(card.querySelector('.agent-insurance').value).toFixed(2),
                    healthcare: parseFloat(card.querySelector('.agent-healthcare').value).toFixed(2),
                    repair: parseFloat(card.querySelector('.agent-repair').value).toFixed(2)
                };
                
                // Validate agent type is not empty
                if (!agent.agent_type.trim()) {
                    showNotification('Please provide a name for all agent types.', 'error');
                    return;
                }
                
                agents.push(agent);
            }
            
            // Check for duplicate agent types
            const agentTypes = agents.map(a => a.agent_type);
            const duplicates = agentTypes.filter((item, index) => agentTypes.indexOf(item) !== index);
            if (duplicates.length > 0) {
                showNotification('Duplicate agent types found: ' + duplicates.join(', '), 'error');
                return;
            }
            
            try {
                const result = await apiCall('/api/agents', 'POST', agents);
                if (result.success) {
                    showNotification('Agents saved successfully!', 'success');
                } else {
                    showNotification('Failed to save agents: ' + result.message, 'error');
                }
            } catch (error) {
                showNotification('Error saving agents: ' + error.message, 'error');
            }
        }
    </script>
</body>
</html>
