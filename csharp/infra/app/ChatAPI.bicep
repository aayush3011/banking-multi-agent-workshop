param name string
param location string = resourceGroup().location
param tags object = {}
param cosmosDbAccountName string
param identityName string
param openAIName string
param containerRegistryName string
param containerAppsEnvironmentId string
param applicationInsightsName string
param exists bool
param envSettings array = []

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2022-02-01-preview' existing = {
  name: containerRegistryName
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: applicationInsightsName
}

resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(subscription().id, resourceGroup().id, identity.id, 'acrPullRole')
  properties: {
    roleDefinitionId:  subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalType: 'ServicePrincipal'
    principalId: identity.properties.principalId
  }
}


resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: openAIName
}

// Role Assignment for Cognitive Services User
resource cognitiveServicesRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(identity.id, openAi.id, 'cognitive-services-user')  // Unique GUID for role assignment
  scope: openAi
  dependsOn: [ identity, openAi ]  // Ensure resources exist before assigning the role
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908')  // Cognitive Services User Role ID
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}


resource cosmosDb 'Microsoft.DocumentDB/databaseAccounts@2023-11-15' existing = {
  name: cosmosDbAccountName
}

resource cosmosAccessRole 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2023-11-15' = {
  name: guid('00000000-0000-0000-0000-000000000002', identity.id, cosmosDb.id)
  parent: cosmosDb
  properties: {
    principalId: identity.properties.principalId
    roleDefinitionId: resourceId('Microsoft.DocumentDB/databaseAccounts/sqlRoleDefinitions', cosmosDb.name, '00000000-0000-0000-0000-000000000002')
    scope: cosmosDb.id
  }
}

module fetchLatestImage '../modules/fetch-container-image.bicep' = {
  name: '${name}-fetch-image'
  params: {
    exists: exists
    name: name
  }
}


resource app 'Microsoft.App/containerApps@2024-02-02-preview' = {
  name: name
  location: location
  tags: union(tags, {'azd-service-name': 'ChatServiceWebApi' })
  dependsOn: [acrPullRole,cognitiveServicesRoleAssignment]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      ingress:  {
        external: true
        targetPort: 8080
        transport: 'auto'
      }
      registries: [
        {
          server: '${containerRegistryName}.azurecr.io'
          identity: identity.id
        }
      ]
	  activeRevisionsMode: 'Single' // Ensures only one active revision at a time
    }
    template: {
      containers: [
        {
          image: fetchLatestImage.outputs.?containers[?0].image ?? 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          name: 'main'
          env: union(
            [
              {
                name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
                value: applicationInsights.properties.connectionString
              }
              {
                name: 'PORT'
                value: '8080'
              }
			  {
				name: 'SemanticKernelServiceSettings__AzureOpenAISettings__UserAssignedIdentityClientID'
				value: identity.properties.clientID
			  }
			  {
				name: 'CosmosDBSettings__UserAssignedIdentityClientID'
				value: identity.properties.clientID
			  }
			  {
				name: 'BankingCosmosDBSettings__UserAssignedIdentityClientID'
				value: identity.properties.clientID
			  }
            ],
            envSettings
          )
          resources: {
            cpu: json('1.0')
            memory: '2.0Gi'
          }  
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
      }
    }
  }
}

output name string = app.name
output uri string = 'https://${app.properties.configuration.ingress.fqdn}'
output id string = app.id
output identity string = identity.properties.principalId
