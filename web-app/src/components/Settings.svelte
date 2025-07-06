<script lang="ts">
  export let agentType: string;
  export let agentPlayer: 'X' | 'O';
  export let availableAgentTypes: string[] = [];
  export let onAgentTypeChange: (type: string) => void = () => {};
  export let onAgentPlayerChange: (player: 'X' | 'O') => void = () => {};
  export let showHistory: boolean = false;
  export let onShowHistoryChange: (show: boolean) => void = () => {};
  export let showValueFunction = false;
  export let onShowValueFunctionChange: (show: boolean) => void = () => {};
</script>

<div class="flex flex-col gap-6 p-2">
  <div>
    <div class="text-xl mb-2 text-white">Agent Type</div>
    <div class="flex gap-4">
      {#each availableAgentTypes as type}
        <button
          type="button"
          class="text-white text-md focus:outline-none transition-transform duration-150 ease-in-out hover:scale-110 {agentType === type ? 'underline font-bold animate-pulse' : ''}"
          aria-current={agentType === type}
          on:click={() => onAgentTypeChange(type)}
        >
          {type}
        </button>
      {/each}
    </div>
  </div>
  <div>
    <div class="text-lg font-semibold mb-2 text-white">Agent Plays As</div>
    <div class="flex gap-4">
      <button
        type="button"
        class="text-white focus:outline-none transition-transform duration-150 ease-in-out {agentPlayer === 'X' ? 'underline font-bold hover:scale-110' : ''}"
        aria-current={agentPlayer === 'X'}
        on:click={() => onAgentPlayerChange('X')}
      >
        X (First)
      </button>
      <button
        type="button"
        class="text-white focus:outline-none transition-transform duration-150 ease-in-out {agentPlayer === 'O' ? 'underline font-bold hover:scale-110' : ''}"
        aria-current={agentPlayer === 'O'}
        on:click={() => onAgentPlayerChange('O')}
      >
        O (Second)
      </button>
    </div>
  </div>
  <div class="mt-4 flex items-center gap-2">
    <input type="checkbox" id="show-value-fn" bind:checked={showValueFunction} on:change={() => onShowValueFunctionChange(showValueFunction)} />
    <label for="show-value-fn" class="text-sm cursor-pointer">Show Value Function on Board</label>
  </div>
</div> 